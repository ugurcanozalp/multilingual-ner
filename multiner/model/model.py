import torch.nn as nn
import torch
import os
import json
from typing import Dict,List,Tuple
from transformers import XLMRobertaModel, XLMRobertaConfig
from multiner.model.subtoken_to_token import SubtokenToToken
from multiner.model.head import NerHead

class MultiNer(nn.Module):

    def __init__(self, n_labels:int, roberta_path:str, load_backbone:bool=False):
        """Multilingual NER model using CRF predictions and XLM-Roberta model as backbone
        """
        super(MultiNer,self).__init__()
        if load_backbone:
            self.roberta = XLMRobertaModel.from_pretrained(roberta_path)
        else:
            cfg = XLMRobertaConfig.from_pretrained(roberta_path)
            self.roberta = XLMRobertaModel(cfg)
        self.ner = NerHead(self.roberta.config.hidden_size, n_labels)

    def forward(self, input_ids:torch.Tensor, attention_mask:torch.Tensor,
            token_type_ids:torch.Tensor, mask:torch.Tensor) -> torch.Tensor:
        """Evaluate word logits from subword sequence
        """
        embedding = self.roberta(input_ids=input_ids,
            attention_mask=attention_mask, token_type_ids=token_type_ids)[0]
        logits, pad_mask = self.ner(embedding, mask)
        return logits, pad_mask

    @torch.no_grad()
    def predict(self, inputs:Tuple[torch.Tensor]) -> torch.Tensor:
        """End-to-end prediction from subword to word sequence predictions
        """
        input_ids, attention_mask, token_type_ids, mask = inputs
        logits, pad_mask = self(input_ids, attention_mask, token_type_ids, mask)
        return self.ner.decode(logits), pad_mask

    def decode(self, logits):
        """Decode logits using CRF weights 
        """
        return self.ner.decode(logits) 

    def eval_loss(self, logits, targets, pad_mask):
        """Calculate CRF Loss from logits and targets for words
        """
        return self.ner.eval_loss(logits, targets, pad_mask)

    def freeze_roberta(self, n_freeze:int=6):
        """Determine how many first layers will be frozen? Calling this function
        freezes word embedding layer anyway, so use accordingly.
        """
        for param in self.roberta.parameters():
            param.requires_grad = False

        for param in self.roberta.encoder.layer[n_freeze:].parameters():
            param.requires_grad = True

