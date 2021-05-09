# Multilingual Named Entity Recognition
Multilingual Named Entity Recognition by XLM-Roberta model with Conditional Random Fields, using zero shot learning.

## Installation
```
conda create --name mner python=3.8.8
git clone https://github.com/ugurcanozalp/multilingual-ner
cd multilingual-ner
pip install -e .
```

## Inference
For inference, you should download the trained model into a folder (for below example, 
![ner_models/gold_model](/ner_models/gold_model) folder)
```python
import multiner
import pprint
ner = multiner.MultiNerInference.load_model("ner_models/gold_model") # Load pretrained model.
text = """World War II or the Second World War, often abbreviated as WWII or WW2, was a global war that lasted from 1939 to 1945. It involved the vast majority of the world's countries—including all the great powers—forming two opposing military alliances: the Allies and the Axis. In a state of total war, directly involving more than 100 million personnel from more than 30 countries, the major participants threw their entire economic, industrial, and scientific capabilities behind the war effort, blurring the distinction between civilian and military resources. World War II was the deadliest conflict in human history, resulting in 70 to 85 million fatalities, with more civilians than military personnel killed. Tens of millions of people died due to genocides (including the Holocaust), premeditated death from starvation, massacres, and disease. Aircraft played a major role in the conflict, including in strategic bombing of population centres, the development of nuclear weapons, and the only two uses of such in war. """
result = ner(text)
pprint.pprint(result)
```

## Training
Pytorch Lightning is used for training. Place data files into data folder. Then run the following command.
```bash
python train.py --train_path "data/train.txt" --val_path "data/dev.txt" --test_path "data/test.txt" --tags_path "data/tags.txt" --gpus 1
```
For testing the model, run the following command.
```bash
python test.py --test_path "data/test.txt" --tags_path "data/tags.txt" --gpus 1
```

## Pretrained model
[!Download](https://drive.google.com/drive/folders/1JMNN9TJWd2oPAl8db1PX-VvXmZMw9h0z?usp=sharing) model (`model.pt`) and tag (`tags.txt`) file from following link. Place it to ![ner_models/gold_model](/ner_models/gold_model) folder. 

## Serving model with TorchServe

Assuming the model files are at ner_models folder, run the following command first.
```bash
torch-model-archiver --model-name multiner --version 1.0 --model-file multiner/model/model.py --serialized-file ner_models/gold/model.pt --export-path model_store/ --extra-files "ner_models/gold/tags.txt,ner_models/xlm-roberta-base/config.json,ner_models/xlm-roberta-base/tokenizer.json,ner_models/xlm-roberta-base/sentencepiece.bpe.model,multiner.zip" --handler multiner/infer.py
```

Then zip the multiner folder to multiner.zip, then add it to created mar file. Lastly create a folder named model_store and move the mar file there. Lastly, run the following command.
```bash
torchserve --start --ncs --model-store model_store/ --models multiner.mar
```

## Onnx Runtime Inference

If you want to use onnx runtime, place torch model into ![ner_models/gold_model](/ner_models/gold_model). Then run the **export2onnx.py** script. Then, you do inference as follows.

```python
import multiner
import pprint
ner = multiner.MultiNerInferenceONNX("ner_models/gold_model") # Load pretrained model.
text = """World War II or the Second World War, often abbreviated as WWII or WW2, was a global war that lasted from 1939 to 1945. It involved the vast majority of the world's countries—including all the great powers—forming two opposing military alliances: the Allies and the Axis. In a state of total war, directly involving more than 100 million personnel from more than 30 countries, the major participants threw their entire economic, industrial, and scientific capabilities behind the war effort, blurring the distinction between civilian and military resources. World War II was the deadliest conflict in human history, resulting in 70 to 85 million fatalities, with more civilians than military personnel killed. Tens of millions of people died due to genocides (including the Holocaust), premeditated death from starvation, massacres, and disease. Aircraft played a major role in the conflict, including in strategic bombing of population centres, the development of nuclear weapons, and the only two uses of such in war. """
result = ner(text)
pprint.pprint(result)
```
