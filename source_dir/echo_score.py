import json
import numpy as np
import os
import nltk
import multiner

def init():
	nltk.download("punkt")
	global ner
	ner = multiner.MultiNerInferONNX(os.getenv("AZUREML_MODEL_DIR"), model_name="optimized.onnx") # Load pretrained model.

def run(data):
    return ner(data['text'])
