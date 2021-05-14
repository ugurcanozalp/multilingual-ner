from typing import Union,Dict,List,Tuple,Any
from argparse import ArgumentParser
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from multiner.model import XLMRobertaNer
from multiner.utils import NerDataset
from seqeval.metrics import accuracy_score, precision_score, recall_score, f1_score
from seqeval.scheme import IOB2

class MultiNer(pl.LightningModule):
    def __init__(self, 
        learning_rate: float=2e-5,
        weight_decay: float=0.0,
        batch_size: int=16,
        freeze_layers: int=8,
        tags_path: os.PathLike=None,
        train_path: os.PathLike=None,
        val_path: os.PathLike=None,
        test_path: os.PathLike=None,
        roberta_path: Union[os.PathLike, str]="xlm-roberta-base",
        pretrained_path: Union[os.PathLike, None]=None,
        *args, **kwargs
    ):
        super(MultiNerTrainer,self).__init__()
        self.save_hyperparameters('learning_rate', 'weight_decay', 'batch_size')
        self.train_path, self.val_path, self.test_path = train_path, val_path, test_path

        self.tags = []
        if tags_path is not None:
            with open(tags_path) as f:
                for line in f:
                    self.tags.append(line.strip())

        self.model = XLMRobertaNer(n_labels=len(self.tags), roberta_path=roberta_path, load_backbone=True)
        if pretrained_path is not None:
            self.model.load_state_dict(torch.load(pretrained_path))
        self.model.freeze_roberta(freeze_layers)

    def forward(self, *args, **kwargs):
        return self.model.forward(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        (input_ids, attention_mask, token_type_ids, mask), labels = batch
        logits, pad_mask = self.model(input_ids, attention_mask, token_type_ids, mask)
        labels = labels[:, :logits.shape[1]]
        loss = self.model.eval_loss(logits, labels, pad_mask)
        preds_tag_idx = self.model.decode(logits)
        preds_tag = [[self.tags[s.item()] for m, s in zip(mask, sample) if m] for mask, sample in zip(pad_mask, preds_tag_idx)]
        labels_tag = [[self.tags[s.item()] for m, s in zip(mask, sample) if m] for mask, sample in zip(pad_mask, labels)]
        tensorboard_logs = {'batch_loss': loss}
        for metric, value in tensorboard_logs.items():
            self.log(metric, value, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        (input_ids, attention_mask, token_type_ids, mask), labels = batch
        logits, pad_mask = self.model(input_ids, attention_mask, token_type_ids, mask)
        labels = labels[:, :logits.shape[1]]
        loss = self.model.eval_loss(logits, labels, pad_mask)
        preds_tag_idx = self.model.decode(logits)
        preds_tag = [[self.tags[s.item()] for m, s in zip(mask, sample) if m] for mask, sample in zip(pad_mask, preds_tag_idx)]
        labels_tag = [[self.tags[s.item()] for m, s in zip(mask, sample) if m] for mask, sample in zip(pad_mask, labels)]
        tensorboard_logs = {'batch_loss': loss}
        for metric, value in tensorboard_logs.items():
            self.log(metric, value, prog_bar=True)
        return {'loss': loss, "preds": preds_tag, "labels": labels_tag}

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        (input_ids, attention_mask, token_type_ids, mask), labels = batch
        logits, pad_mask = self.model(input_ids, attention_mask, token_type_ids, mask)
        labels = labels[:, :logits.shape[1]]
        loss = self.model.eval_loss(logits, labels, pad_mask)
        preds_tag_idx = self.model.decode(logits)
        preds_tag = [[self.tags[s.item()] for m, s in zip(mask, sample) if m] for mask, sample in zip(pad_mask, preds_tag_idx)]
        labels_tag = [[self.tags[s.item()] for m, s in zip(mask, sample) if m] for mask, sample in zip(pad_mask, labels)]
        tensorboard_logs = {'batch_loss': loss}
        for metric, value in tensorboard_logs.items():
            self.log(metric, value, prog_bar=True)
        return {'loss': loss, "preds": preds_tag, "labels": labels_tag}
   
    def configure_optimizers(self):
        no_decay_keywords = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n,p in self.model.named_parameters() if not any(nd in n for nd in no_decay_keywords)],
                "weight_decay_rate": self.hparams.weight_decay,
                "lr": self.hparams.learning_rate,
            },
            {
                "params": [p for n,p in self.model.named_parameters() if any(nd in n for nd in no_decay_keywords)],
                "weight_decay_rate": 0,
                "lr": self.hparams.learning_rate,
            }
        ]
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters)
        return optimizer

    def train_dataloader(self):
        dataset = NerDataset(self.train_path, label_tags = self.tags, default_label=0, to_device=self.device)
        return DataLoader(dataset, drop_last=False, shuffle=True, batch_size=self.hparams.batch_size, 
            worker_init_fn=np.random.seed(0))

    def val_dataloader(self):
        dataset = NerDataset(self.val_path, label_tags = self.tags, default_label=0, to_device=self.device)
        return DataLoader(dataset, drop_last=False, shuffle=False, batch_size=self.hparams.batch_size, 
            worker_init_fn=np.random.seed(0))

    def test_dataloader(self):
        dataset = NerDataset(self.test_path, label_tags = self.tags, default_label=0, to_device=self.device)
        return DataLoader(dataset, drop_last=False, shuffle=False, batch_size=self.hparams.batch_size, 
            worker_init_fn=np.random.seed(0))

    def validation_epoch_end(self, outputs):
        preds = sum([x['preds'] for x in outputs], [])
        labels = sum([x['labels'] for x in outputs], [])
        loss = torch.stack([x['loss'] for x in outputs]).mean()
        acc = accuracy_score(labels, preds)
        precision = precision_score(labels, preds, mode='strict', scheme=IOB2, average='micro')
        recall = recall_score(labels, preds, mode='strict', scheme=IOB2, average='micro')
        f1 = f1_score(labels, preds, mode='strict', scheme=IOB2, average='micro')
        tensorboard_logs = {'val_loss': loss, 'val_accuracy': acc, 'val_precision': precision, 'val_recall': recall, 'val_F1': f1}
        for metric, value in tensorboard_logs.items():
            self.log(metric, value, prog_bar=True)

    def test_epoch_end(self, outputs):
        preds = sum([x['preds'] for x in outputs], [])
        labels = sum([x['labels'] for x in outputs], [])
        loss = torch.stack([x['loss'] for x in outputs]).mean()
        acc = accuracy_score(labels, preds)
        precision = precision_score(labels, preds, mode='strict', scheme=IOB2, average='micro')
        recall = recall_score(labels, preds, mode='strict', scheme=IOB2, average='micro')
        f1 = f1_score(labels, preds, mode='strict', scheme=IOB2, average='micro')
        tensorboard_logs = {'test_loss': loss, 'test_accuracy': acc, 'test_precision': precision, 'test_recall': recall, 'test_F1': f1}
        for metric, value in tensorboard_logs.items():
            self.log(metric, value, prog_bar=True)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--learning_rate', type=float, default=2e-5)
        parser.add_argument('--weight_decay', type=float, default=1e-4)
        parser.add_argument('--batch_size', type=int, default=16)
        parser.add_argument('--freeze_layers', type=int, default=8)
        parser.add_argument('--train_path', type=str, default=None)
        parser.add_argument('--val_path', type=str, default=None)
        parser.add_argument('--test_path', type=str, default=None)
        parser.add_argument('--tags_path', type=str, default=None)
        parser.add_argument('--roberta_path', type=str, default='xlm-roberta-base')
        parser.add_argument('--pretrained_path', type=str, default=None)
        return parser