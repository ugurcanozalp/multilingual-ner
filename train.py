import os
from argparse import ArgumentParser
import torch
import pytorch_lightning as pl
from multiner import MultiNerTrainer

def train(args):
    dict_args = vars(args)
    plmodel = MultiNerTrainer(**dict_args)
    pytorch_trainable_params = sum(p.numel() for p in plmodel.model.parameters() if p.requires_grad)
    pytorch_total_params = sum(p.numel() for p in plmodel.model.parameters())
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(plmodel)
    result = trainer.test(plmodel)
    print(result)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser = MultiNerTrainer.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    train(args)  

# python train.py --learning_rate 2e-5 --weight_decay 0 --batch_size 8 --accumulate_grad_batches 2 --gradient_clip_val 1.0 --max_epochs 10 --min_epochs 3  --train_path "data/train.txt" --val_path "data/dev.txt" --test_path "data/test.txt" --tags_path "data/tags.txt" --gpus 1