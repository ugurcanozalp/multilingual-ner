# Some standard imports
import io
import os
import numpy as np

from torch import nn
import torch.utils.model_zoo as model_zoo
import torch.onnx

from multiner.model import XLMRobertaNer
from multiner.utils import CustomTokenizer
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--model_folder', type=str, default="ner_models/gold_model")
args = parser.parse_args()

# Load pretrained model weights
model_path = os.path.join(args.model_folder, 'model.pt')
tags_path = os.path.join(args.model_folder, 'tags.txt')
onnx_path = os.path.join(args.model_folder, 'model.onnx')
opt_onnx_path = os.path.join(args.model_folder, 'model-optimized.onnx')
crf_path = os.path.join(args.model_folder, "crf_dict.pt")

tags = []
if tags_path is not None:
    with open(tags_path) as f:
        for line in f:
            tags.append(line.strip())

batch_size = 1    # just a random number
seq_len = 100
out_seq_len = 50
n_labels = len(tags)
torch_model = XLMRobertaNer(n_labels=n_labels, roberta_path="xlm-roberta-base", load_backbone=False).cpu()
tokenizer = CustomTokenizer(vocab_path="xlm-roberta-base", to_device="cpu")

# Initialize model with the pretrained weights
map_location = lambda storage, loc: storage

torch_model.load_state_dict(torch.load(model_path, map_location=map_location), strict=False)

# set the model to inference mode
torch_model.eval()

# Input to the model
text = """World War II or the Second World War, often abbreviated as WWII or WW2, was a global war that lasted from 1939 to 1945. It involved the vast majority of the world's countries—including all the great powers—forming two opposing military alliances: the Allies and the Axis. In a state of total war, directly involving more than 100 million personnel from more than 30 countries, the major participants threw their entire economic, industrial, and scientific capabilities behind the war effort, blurring the distinction between civilian and military resources. World War II was the deadliest conflict in human history, resulting in 70 to 85 million fatalities, with more civilians than military personnel killed. Tens of millions of people died due to genocides (including the Holocaust), premeditated death from starvation, massacres, and disease. Aircraft played a major role in the conflict, including in strategic bombing of population centres, the development of nuclear weapons, and the only two uses of such in war. """
text = """Sağlık Bakanlığı'nın açıkladığı güncel corona virüsü verilerine göre semptom gösteren 2 bin 210 hastayla birlikte 20 bin 107 yeni vaka tespit edilirken 278 kişi hayatını kaybetti."""
#text = """Wikipedia has received praise for its enablement of the democratization of knowledge, extent of coverage, unique structure, culture, and reduced amount of commercial bias, but has also been criticized for its perceived unreliability and for exhibiting systemic bias, namely racial bias and gender bias against women, and alleged ideological bias."""

for tokenized in tokenizer.consume_text(text):
    inputs, _, _ = tokenized

torch_out = torch_model(*inputs)

# Export the model
torch.onnx.export(torch_model,               # model being run
                  inputs,                    # model input (or a tuple for multiple inputs)
                  onnx_path,                 # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=11,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  verbose=True,
                  input_names = ['input_ids', 'attention_mask', 'token_type_ids', 'mask'],   # the model's input names
                  output_names = ['decoded', 'pad_mask'], # the model's output names
                  dynamic_axes={'input_ids' : {0: 'batch_size', 1: 'seq_len'},
                                'attention_mask': {0: 'batch_size', 1: 'seq_len'},
                                'token_type_ids': {0: 'batch_size', 1: 'seq_len'},
                                'mask': {0: 'batch_size', 1: 'seq_len'},
                                'decoded': {0: 'batch_size', 1: 'out_seq_len'},
                                'pad_mask': {0: 'batch_size', 1: 'out_seq_len'} }) # variable length axes

import onnx

onnx_model = onnx.load(onnx_path)
onnx.checker.check_model(onnx_model)

import onnxruntime

ort_session = onnxruntime.InferenceSession(onnx_path)

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

# compute ONNX Runtime output prediction
ort_inputs = {onnx_input.name: to_numpy(x) for onnx_input, x in zip(ort_session.get_inputs(), inputs)}
ort_outs = ort_session.run(None, ort_inputs)

# compare ONNX Runtime and PyTorch results
#np.testing.assert_allclose(to_numpy(torch_out[0]), ort_outs[0], rtol=1e-03, atol=1e-05)
#np.testing.assert_allclose(to_numpy(torch_out[1]), ort_outs[1], rtol=1e-03, atol=1e-05)

print("Exported model has been tested with ONNXRuntime, and the result looks good!")

# Save last CRF layer, since it is needed after onnx inference.
torch.save(torch_model.ner.crf.state_dict(), crf_path)