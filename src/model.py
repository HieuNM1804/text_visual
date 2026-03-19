import copy
import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.nn import functional as F

from torchmetrics.functional import retrieval_average_precision, retrieval_precision

from src.coprompt import MultiModalPromptLearner, Adapter, TextEncoder
from src.utils import load_clip_to_cpu, get_all_categories
from src.losses import loss_fn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def freeze_model(m):
    m.requires_grad_(False)
    
def freeze_all_but_bn(m):
    if not isinstance(m, torch.nn.LayerNorm):
        if hasattr(m, 'weight') and m.weight is not None:
            m.weight.requires_grad_(False)
        if hasattr(m, 'bias') and m.bias is not None:
            m.bias.requires_grad_(False)
            
class CustomCLIP(nn.Module):
    def __init__(
        self, cfg, clip_model, clip_model_distill
    ):
        super().__init__()
        self.cfg = cfg
        clip_model.apply(freeze_all_but_bn)
        clip_model_distill.apply(freeze_all_but_bn)
        self.dtype = clip_model.dtype
        self.prompt_learner_photo = MultiModalPromptLearner(cfg, clip_model_distill, type='photo')
        self.prompt_learner_sketch = MultiModalPromptLearner(cfg, clip_model_distill, type='sketch')
        
        self.ph_encoder = copy.deepcopy(clip_model.visual)
        self.sk_encoder = copy.deepcopy(clip_model.visual)
        self.text_encoder = TextEncoder(clip_model_distill, cfg)
        self.logit_scale = clip_model.logit_scale
        self.text_stage_depth = cfg.text_stage_depth
        self.visual_stage_depth = cfg.visual_stage_depth
        
        self.adapter_photo = Adapter(512, 4).to(clip_model.dtype)
        self.adapter_text = Adapter(512, 4).to(clip_model.dtype)
        self.text_to_visual_bridge = nn.Linear(512, 768).to(clip_model.dtype)
        self.late_text_to_visual_gate = nn.Parameter(torch.tensor(-1.5))
        
        self.model_distill = clip_model_distill
        self.image_adapter_m = 0.1
        self.text_adapter_m = 0.1

    def _gate(self, gate_param, ref_tensor):
        return torch.sigmoid(gate_param).to(dtype=ref_tensor.dtype, device=ref_tensor.device)
    
    def get_logits(self, img_tensor, classnames, type='photo'):
        if type=='photo':
            prompt_learner = self.prompt_learner_photo
            image_encoder = self.ph_encoder
        else:
            image_encoder = self.sk_encoder
            prompt_learner = self.prompt_learner_sketch
        # tokenized_prompts = prompt_learner.tokenized_prompts
        logit_scale = self.logit_scale.exp()
        (
            tokenized_prompts,
            prompts,
            shared_ctx,
            # deep_compound_prompts_text,
            # deep_compound_prompts_vision,
        ) = prompt_learner(classnames)

        text_hidden = self.text_encoder.embed_prompts(prompts)
        total_text_blocks = len(self.text_encoder.resblocks)
        total_visual_blocks = len(image_encoder.transformer.resblocks)
        text_stage_depth = min(max(self.text_stage_depth, 1), total_text_blocks)
        visual_stage_depth = min(max(self.visual_stage_depth, 1), total_visual_blocks)

        shared_ctx = shared_ctx.type(self.dtype)
        # Keep the first visual stage text-free; only the later visual blocks
        # receive text-guided prompt updates.
        visual_hidden = image_encoder.embed_tokens(
            img_tensor.type(self.dtype),
            torch.zeros_like(shared_ctx),
        )
        visual_hidden = image_encoder.run_blocks(visual_hidden, end=visual_stage_depth)

        text_hidden = self.text_encoder.run_blocks(
            text_hidden,
            end=text_stage_depth,
        )
        text_hidden = self.text_encoder.run_blocks(
            text_hidden,
            start=text_stage_depth,
            end=total_text_blocks,
        )
        text_features = self.text_encoder.finalize_features(text_hidden, tokenized_prompts)

        late_text_ctx = self.text_encoder.extract_prompt_context(text_hidden).mean(dim=1)
        late_visual_ctx = self.text_to_visual_bridge(late_text_ctx.type(self.dtype))
        visual_stage2_gate = self._gate(self.late_text_to_visual_gate, visual_hidden)
        visual_hidden = image_encoder.run_blocks(
            visual_hidden,
            start=visual_stage_depth,
            end=total_visual_blocks,
            injected_ctx=late_visual_ctx,
            blend=visual_stage2_gate,
        )
        image_features = image_encoder.finalize_tokens(visual_hidden)
        
        x_a = self.adapter_photo(image_features)
        image_features = (
            self.image_adapter_m * x_a + (1 - self.image_adapter_m) * image_features
        )

        x_b = self.adapter_text(text_features)
        text_features = (
            self.text_adapter_m * x_b + (1 - self.text_adapter_m) * text_features
        )

        image_features_normalize = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logits = logit_scale * image_features_normalize @ text_features.t()
        
        return logits, image_features_normalize, image_features, text_features
        
    def forward(self, x, classnames):
        photo_tensor, sk_tensor, photo_aug_tensor, sk_aug_tensor, neg_tensor, label = x
        pos_logits, photo_features_norm, photo_feature, ph_txt_feature = self.get_logits(photo_tensor, classnames)
        sk_logits, sk_feature_norm, sk_feature, sk_txt_feature = self.get_logits(sk_tensor, classnames, type='sketch')
        _, neg_feature, _, _ = self.get_logits(neg_tensor, classnames)
        
        photo_aug_features = self.model_distill.encode_image(photo_aug_tensor)
        sk_aug_features = self.model_distill.encode_image(sk_aug_tensor)
            
        return photo_features_norm, sk_feature_norm, photo_aug_features, sk_aug_features, \
            neg_feature, label, pos_logits, sk_logits, photo_feature, sk_feature, \
                ph_txt_feature, sk_txt_feature
        
    def extract_feature(self, image, classname, type='photo'):
        _, feature, _, _ = self.get_logits(image, classnames=classname, type=type)
        return feature
            
class ZS_SBIR(pl.LightningModule):
    def __init__(self, args, classname):
        super(ZS_SBIR, self).__init__()
        self.args = args
        self.classname = classname
        clip_model = load_clip_to_cpu(args)
        
        design_details = {
            "trainer": "CoOp",
            "vision_depth": 0,
            "language_depth": 0,
            "vision_ctx": 0,
            "language_ctx": 0,
        }
        clip_model_distill = load_clip_to_cpu(args, design_details=design_details)
        
        self.distance_fn = lambda x, y: F.cosine_similarity(x, y)
        self.best_metric = 1e-3
        
        self.model = CustomCLIP(cfg=args, clip_model=clip_model, clip_model_distill=clip_model_distill)
    
        self.val_step_outputs_sk = []
        self.val_step_outputs_ph = []
        
        
    def configure_optimizers(self):
        optimizer = torch.optim.SGD(params=self.model.parameters(), lr=self.args.lr, weight_decay=1e-3, momentum=0.9)
        
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer=optimizer,
            step_size=5,
            gamma=0.1
        )
        
        return [optimizer] , [scheduler]
    
    def forward(self, data, classname):
        return self.model(data, classname)
    
    def training_step(self, batch, batch_idx):
        classname = get_all_categories(self.args)
        features = self.forward(batch, classname)
        
        loss = loss_fn(self.args, self.model, features=features, mode='train')
        self.log('train_loss', loss, on_step=False, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx, dataloader_idx):
        classnames = get_all_categories(self.args, mode="test")
        image_tensor, label = batch
        if dataloader_idx == 0:
            feat = self.model.extract_feature(image_tensor, classname=classnames, type='sketch')
            self.val_step_outputs_sk.append((feat, label))
        else:
            feat = self.model.extract_feature(image_tensor, classname=classnames, type='photo')
            self.val_step_outputs_ph.append((feat, label))
    
    def on_validation_epoch_end(self):
        query_len = len(self.val_step_outputs_sk)
        gallery_len = len(self.val_step_outputs_ph)
        
        query_feat_all = torch.cat([self.val_step_outputs_sk[i][0] for i in range(query_len)])
        gallery_feat_all = torch.cat([self.val_step_outputs_ph[i][0] for i in range(gallery_len)])
        
        all_sketch_category = np.array(sum([list(self.val_step_outputs_sk[i][1].detach().cpu().numpy()) for i in range(query_len)], []))
        all_photo_category = np.array(sum([list(self.val_step_outputs_ph[i][1].detach().cpu().numpy()) for i in range(gallery_len)], []))
        
        ## mAP category-level SBIR Metrics
        gallery = gallery_feat_all
        ap = torch.zeros(len(query_feat_all))
        precision = torch.zeros(len(query_feat_all))
        if self.args.dataset == "sketchy_2":
            map_k = 200
            p_k = 200
        else:
            map_k = 0
            if self.args.dataset == "quickdraw":
                p_k = 200
            else:
                p_k = 100
                
        for idx, sk_feat in enumerate(query_feat_all):
            category = all_sketch_category[idx]
            distance = self.distance_fn(sk_feat.unsqueeze(0), gallery)
            target = torch.zeros(len(gallery), dtype=torch.bool, device=device)
            target[np.where(all_photo_category == category)] = True
            
            if map_k != 0:
                top_k_actual = min(map_k, len(gallery)) 
                ap[idx] = retrieval_average_precision(distance.cpu(), target.cpu(), top_k=top_k_actual)
            else: 
                ap[idx] = retrieval_average_precision(distance.cpu(), target.cpu())
                
            precision[idx] = retrieval_precision(distance.cpu(), target.cpu(), top_k=p_k)
            
            
        mAP = torch.mean(ap)
        precision = torch.mean(precision)
        self.log("mAP", mAP, on_step=False, on_epoch=True)
        if self.global_step > 0:
            self.best_metric = self.best_metric if  (self.best_metric > mAP.item()) else mAP.item()
        
        if map_k != 0:
            print('mAP@{}: {}, P@{}: {}, Best mAP: {}'.format(map_k, mAP.item(), p_k, precision, self.best_metric))
        else:
            print('mAP@all: {}, P@{}: {}, Best mAP: {}'.format(mAP.item(), p_k, precision, self.best_metric))
        train_loss = self.trainer.callback_metrics.get("train_loss", None)

        if train_loss is not None:
            print(f"Train loss (epoch avg): {train_loss.item():.6f}")
        self.val_step_outputs_sk.clear()
        self.val_step_outputs_ph.clear()
