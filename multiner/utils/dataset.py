from typing import Union, List, Any
import torch
import numpy as np
from torch.utils.data import Dataset
from multiner.utils.custom_tokenizer import CustomTokenizer

class NerDataset(Dataset):
    '''This dataset is loader for Named Entity Recognition dataset.
    '''

    def __init__(self, data_path:str, label_tags: List[str], vocab_path:str="xlm-roberta-base", 
            default_label=0, max_length:int=512, to_device="cpu"):
        self.tokenizer = CustomTokenizer(vocab_path=vocab_path, to_device=to_device)
        self.label_tags = label_tags        
        self.name_to_label = {x: i for i, x in enumerate(self.label_tags)}
        self.default_label = default_label
        self.max_length = max_length

        with open(data_path,'r') as f:
            data_text = f.read()

        self.data = []
        for sentence in filter(lambda x: len(x)>2, data_text.split('\n\n')):
            sample = []
            for wordline in sentence.split('\n'):
                if wordline=='':
                    continue
                word, label = wordline.split('\t')
                sample.append((word, label))
            self.data.append(sample)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        words, labels = list(zip(*item))
        
        labels_idx = [self.name_to_label.get(x, self.default_label) for x in labels]  
        y = torch.tensor(labels_idx, dtype=torch.long)
        diff = self.max_length - y.shape[-1]
        y = torch.nn.functional.pad(y, (0, diff), value=self.default_label)

        X = self.tokenizer.tokenize_words_batch(list(words), pad_to=self.max_length)

        return X, y 
