import os
import copy
import torch
import torch.nn as nn
from torch import Tensor, tensor

from torch.nn import functional as F

from clip import clip
from src.data_config import UNSEEN_CLASSES

def retrieval_average_precision(preds, target, top_k = None):
    top_k = top_k or preds.shape[-1]
    if not isinstance(top_k, int) and top_k <= 0:
        raise ValueError(f"Argument ``top_k`` has to be a positive integer or None, but got {top_k}.")

    target = target[preds.topk(min(top_k, preds.shape[-1]), sorted=True, dim=-1)[1]]

    if not target.sum():
        return tensor(0.0, device=preds.device)

    positions = torch.arange(1, len(target) + 1, device=target.device, dtype=torch.float32)[target > 0]
    return torch.div((torch.arange(len(positions), device=positions.device, dtype=torch.float32) + 1), positions).mean()


def get_all_categories(args, mode="train"):
    all_categories = os.listdir(os.path.join(args.root, 'sketch'))
    unseen_classes = UNSEEN_CLASSES[args.dataset]
    if '.ipynb_checkpoints' in all_categories:
        all_categories.remove('.ipynb_checkpoints')
    if mode=="train":
        all_categories = sorted(list(set(all_categories) - set(unseen_classes)))
    else:
        all_categories = sorted(unseen_classes)
    return all_categories

def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def init_weight(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
            
def load_clip_to_cpu(cfg, design_details=None):
    backbone_name = cfg.backbone
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    if design_details is None:
        design_details = {
            "trainer": "CoPrompt",
            "vision_depth": 0,
            "language_depth": 0,
            "vision_ctx": 0,
            "language_ctx": 0,
            "maple_length": cfg.n_ctx,
        }
    model = clip.build_model(state_dict or model.state_dict(), design_details)

    return model