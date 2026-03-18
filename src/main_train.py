import os
import torch
import numpy as np
import random
import argparse
from torch.utils.data import DataLoader, Subset
from pytorch_lightning import Trainer 
from pytorch_lightning.loggers import TensorBoardLogger 
from pytorch_lightning.callbacks import ModelCheckpoint 

from src.sketchy_dataset import TrainDataset, ValidDataset
from src.model import ZS_SBIR
from src.utils import get_all_categories

def get_datasets(opts):
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
    train_dataset = TrainDataset(opts)
    val_sketch = ValidDataset(opts, mode='sketch')
    val_photo = ValidDataset(opts)

    train_loader = DataLoader(dataset=train_dataset, batch_size=opts.batch_size, num_workers=opts.workers, shuffle=True)
    val_sketch_loader = DataLoader(dataset=val_sketch, batch_size=opts.test_batch_size, num_workers=opts.workers, shuffle=False)
    val_photo_loader = DataLoader(dataset=val_photo, batch_size=opts.test_batch_size, num_workers=opts.workers, shuffle=False)

    return train_loader, val_sketch_loader, val_photo_loader

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="../datasets/tuberlin", help="path to dataset")
    parser.add_argument("--ckpt_path", type=str, default="", help="path to dataset")
    parser.add_argument("--dataset", type=str, default="tuberlin", help="type of dataset")
    parser.add_argument("--output_dir", type=str, default="", help="output directory")
    parser.add_argument("--backbone", type=str, default="ViT-B/32")
    parser.add_argument("--n_ctx", type=int, default=2)
    parser.add_argument("--img_ctx", type=int, default=2)
    parser.add_argument("--max_size", type=int, default=224)
    parser.add_argument("--prompt_depth", type=int, default=12)
    parser.add_argument("--text_stage_depth", type=int, default=6)
    parser.add_argument("--visual_stage_depth", type=int, default=6)
    parser.add_argument("--data_split", type=int, default=-1)
    parser.add_argument("--prec", type=str, default="fp16")
    parser.add_argument("--distill", type=str, default="cosine")
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--alpha", type=float, default=0.8)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--lambd", type=float, default=0.1)
    
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--test_batch_size', type=int, default=1024)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--use_adapt_sk', type=bool, default=True)
    parser.add_argument('--use_adapt_ph', type=bool, default=True)
    parser.add_argument('--use_adapt_txt', type=bool, default=True)
    parser.add_argument('--use_co_sk', type=bool, default=True)
    parser.add_argument('--use_co_ph', type=bool, default=True)
    parser.add_argument('--progress', type=bool, default=False)
    parser.add_argument('--use_subset', type=bool, default=False)
    
    parser.add_argument('--exp_name', type=str, default='Co_prompt')
    
    args = parser.parse_args()
    logger = TensorBoardLogger('tb_logs', name=args.exp_name)
    
    checkpoint_callback = ModelCheckpoint(
        monitor='mAP',
        dirpath='saved_models/%s'%args.exp_name,
        filename="{epoch:02d}-{mAP:.4f}",
        save_top_k=1,
        mode='max',
        save_last=True)
    
    ckpt_path = args.ckpt_path
    if not os.path.exists(ckpt_path):
        ckpt_path = None
    else:
        print ('resuming training from %s'%ckpt_path)

    train_loader, val_sketch_loader, val_photo_loader = get_datasets(args)
    trainer = Trainer(accelerator='gpu', devices=1, 
        min_epochs=1, max_epochs=args.epochs,
        benchmark=True,
        logger=logger,
        check_val_every_n_epoch=1,
        enable_progress_bar=args.progress,
        callbacks=[checkpoint_callback]
    )

    classnames = get_all_categories(args)
 
    if ckpt_path is None:
        model = ZS_SBIR(args=args, classname=classnames)
    else:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        sd = ckpt["state_dict"]

        skip = [
            "model.prompt_learner_photo.token_prefix",
            "model.prompt_learner_photo.token_suffix",
            "model.prompt_learner_sketch.token_prefix",
            "model.prompt_learner_sketch.token_suffix",
        ]
        for k in skip:
            sd.pop(k, None)

        model = ZS_SBIR(args=args, classname=classnames)  # classnames = 220
        missing, unexpected = model.load_state_dict(sd, strict=False)

    trainer.fit(model, train_loader, [val_sketch_loader, val_photo_loader])
