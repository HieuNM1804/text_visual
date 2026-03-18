import copy
import torch
import torch.nn as nn
from torch.nn import functional as F

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from src.data_config import UNSEEN_CLASSES

_tokenizer = _Tokenizer()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TextEncoder(nn.Module):
    def __init__(self, clip_model, args):
        super().__init__()
        self.args = args
        self.resblocks = clip_model.transformer.resblocks
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def embed_prompts(self, prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        return x.permute(1, 0, 2)

    def extract_prompt_context(self, x):
        return x[1:self.args.n_ctx + 1, :, :]

    def apply_prompt_context(self, x, ctx_tokens, blend=None):
        prefix = x[:1, :, :]
        suffix = x[1 + self.args.n_ctx :, :, :]

        if ctx_tokens.dim() == 2:
            ctx_tokens = ctx_tokens.unsqueeze(1).expand(-1, x.shape[1], -1)

        ctx_tokens = ctx_tokens.type_as(x)
        if blend is not None:
            current_ctx = self.extract_prompt_context(x)
            ctx_tokens = torch.lerp(current_ctx, ctx_tokens, blend.to(dtype=x.dtype, device=x.device))

        return torch.cat([prefix, ctx_tokens, suffix], dim=0)

    def run_blocks(self, x, start=0, end=None, injected_ctx=None, blend=None, reinject=False):
        if end is None:
            end = len(self.resblocks)

        for index in range(start, end):
            if injected_ctx is not None and (reinject or index == start):
                x = self.apply_prompt_context(x, injected_ctx, blend=blend)
            x = self.resblocks[index](x)

        return x

    def finalize_features(self, x, tokenized_prompts):
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)
        return x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

    def forward(self, prompts, tokenized_prompts, return_all=False):
        x = self.embed_prompts(prompts)
        # Pass as the list, as nn.sequential cannot process multiple arguments in the forward pass
        
        txt_guided_prompts = []
        for block in self.resblocks:
            x = block(x)
            
            # Capture the learnable text prompt tokens after each text block so
            # they can be reused as the source for layer-wise visual prompts.
            prompt_tok = x[1:self.args.n_ctx + 1,:, :]
            prompt_tok = prompt_tok.permute(1, 0, 2)
            txt_guided_prompts.append(prompt_tok)
            
        x = self.finalize_features(x, tokenized_prompts)

        if return_all:
            return x, txt_guided_prompts
        return x
    
class MultiModalPromptLearner(nn.Module):
    def __init__(self, cfg, clip_model, type='photo'):
        super().__init__()
        self.clip_model = clip_model
        self.cfg = cfg
        n_ctx = cfg.n_ctx
        ctx_init = "a photo/sketch of "
            
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.max_size
        
        self.dropout_layer = nn.Dropout(p=0.1)
        self.compound_prompts_depth = (
            cfg.prompt_depth
        )  # max=12, but will create 11 such shared prompts
        assert (
            cfg_imsize == clip_imsize
        ), f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init and (n_ctx) <= 4:
            # use given words to initialize context vectors
            n_ctx = n_ctx
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            # random initialization
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors)
            prompt_prefix = ctx_init
        
        self.prompt_prefix = prompt_prefix
        self.proj = nn.Linear(ctx_dim, 768)
        
        if dtype == torch.float16:
            self.proj.half()
        self.ctx = nn.Parameter(ctx_vectors)

        # self.compound_prompts_text = nn.ParameterList(
        #     [
        #         nn.Parameter(torch.empty(n_ctx, 512))
        #         for _ in range(self.compound_prompts_depth - 1)
        #     ]
        # )
        # for single_para in self.compound_prompts_text:
        #     nn.init.kaiming_uniform_(single_para)
        
        

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        # self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        # self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

        self.n_ctx = n_ctx
        
    def construct_prompts(self, ctx, prefix, suffix, label=None):
            
        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]

        prompts = torch.cat(
            [
                prefix,  # (dim0, 1, dim)
                ctx,  # (dim0, n_ctx, dim)
                suffix,  # (dim0, *, dim)
            ],
            dim=1,
        )

        return prompts

    def forward(self, classnames, label=None):
        n_cls = len(classnames)
        classnames = [name.replace("_", " ") for name in classnames]
        raw_prompts = [self.prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in raw_prompts])
        with torch.no_grad():
            embedding = self.clip_model.token_embedding(tokenized_prompts.to(device)).type(self.clip_model.dtype)
        
        ctx = self.ctx
        if self.training:
            ctx = self.dropout_layer(ctx)
        
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(n_cls, -1, -1)

        prefix = embedding[:, :1, :]
        suffix = embedding[:, 1 + self.cfg.n_ctx :, :]
        
        prompts = self.construct_prompts(ctx, prefix, suffix, label)
        
        return (
            tokenized_prompts,
            prompts,
            self.proj(self.ctx),
            # self.compound_prompts_text, 
            # visual_deep_prompts,        
        )  # pass here original, as for visual 768 is required

class Adapter(nn.Module):
    def __init__(self, c_in, reduction=4):
        super(Adapter, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(c_in, c_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c_in // reduction, c_in, bias=False),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.fc(x)
        return x
    
