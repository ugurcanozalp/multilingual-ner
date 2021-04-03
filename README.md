# multilingual-ner
Multilingual Named Entity Recognition by XLM-Roberta model with Conditional Random Fields.

## Installation
```
git clone https://github.com/ugurcanozalp/multilingual-ner
cd multilingual-ner
pip install -e .
```

## Inference
For inference, you should download the trained model into a folder (for below example, 
![model_store/gold_model](/model_store/gold_model) folder)
```python
import multiner
import pprint
ner = multiner.MultiNerInference.load_model("model_store/gold_model") # Load pretrained model.
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