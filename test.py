import os
from argparse import ArgumentParser
import torch
import pytorch_lightning as pl
from multiner import MultiNerTrainer

def test(args):
    dict_args = vars(args)
    plmodel = MultiNerTrainer(**dict_args) 
    pytorch_total_params = sum(p.numel() for p in plmodel.model.parameters())
    trainer = pl.Trainer.from_argparse_args(args)
    result = trainer.test(plmodel)
    print(result)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser = MultiNerTrainer.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    test(args)  

# python test.py --batch_size 8 --accumulate_grad_batches 2  --test_path "data/test.txt" --tags_path "data/tags.txt" --gpus 1