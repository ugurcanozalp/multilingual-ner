import torch.nn as nn
import torch
from multiner.model.crf import CRF
from multiner.model.subtoken_to_token import SubtokenToToken

class NerHead(nn.Module):
    def __init__(self, embedding_size, n_labels):
        """Generic NER prediction layer using CRF.
        """
        super(NerHead, self).__init__()
        self.dropout = nn.Dropout(0.1)
        self.output_dense = nn.Linear(embedding_size,n_labels)
        self.crf = CRF(n_labels, batch_first=True)
        self.token_from_subtoken = SubtokenToToken()

    def forward(self, embedding, mask):
        """Evaluate logits from backbone embeddings
        """
        logits = self.output_dense(self.dropout(embedding))
        logits = self.token_from_subtoken(logits, mask)
        pad_mask = self.token_from_subtoken(mask.unsqueeze(-1), mask).squeeze(-1).bool()
        return logits, pad_mask

    def decode(self, logits):
        """Decode logits using CRF weights 
        """
        return self.crf.decode(logits)

    def eval_loss(self, logits, targets, pad_mask):
        """Calculate CRF Loss from logits and targets for words
        """
        mean_log_likelihood = self.crf(logits, targets, pad_mask, reduction='sum').mean()
        return -mean_log_likelihood
