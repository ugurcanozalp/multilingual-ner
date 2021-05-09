# Some standard imports
import io
import numpy as np

from torch import nn
import torch.utils.model_zoo as model_zoo
import torch.onnx

from multiner.model import MultiNer, MultiNerForward
from multiner.utils import CustomTokenizer

# Load pretrained model weights
model_path = 'ner_models/gold_model/model.pt'
tags_path = 'ner_models/gold_model/tags.txt'
tags = []
if tags_path is not None:
    with open(tags_path) as f:
        for line in f:
            tags.append(line.strip())

batch_size = 1    # just a random number
seq_len = 100
out_seq_len = 50
n_labels = len(tags)
torch_model = MultiNerForward(n_labels=n_labels, roberta_path="xlm-roberta-base", load_backbone=False).half().cuda()
tokenizer = CustomTokenizer(vocab_path="xlm-roberta-base", to_device="cuda")

# Initialize model with the pretrained weights
if torch.cuda.is_available():
    map_location = None
else:
    map_location = lambda storage, loc: storage

torch_model.load_state_dict(torch.load(model_path, map_location=map_location), strict=False)

# set the model to inference mode
torch_model.eval()

# Input to the model
text = """World War II or the Second World War, often abbreviated as WWII or WW2, was a global war that lasted from 1939 to 1945. It involved the vast majority of the world's countries—including all the great powers—forming two opposing military alliances: the Allies and the Axis. In a state of total war, directly involving more than 100 million personnel from more than 30 countries, the major participants threw their entire economic, industrial, and scientific capabilities behind the war effort, blurring the distinction between civilian and military resources. World War II was the deadliest conflict in human history, resulting in 70 to 85 million fatalities, with more civilians than military personnel killed. Tens of millions of people died due to genocides (including the Holocaust), premeditated death from starvation, massacres, and disease. Aircraft played a major role in the conflict, including in strategic bombing of population centres, the development of nuclear weapons, and the only two uses of such in war. """

for tokenized in tokenizer.consume_text(text):
    inputs, _, _ = tokenized

torch_out = torch_model(*inputs)

print(torch_out)
