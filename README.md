# Multilingual Named Entity Recognition
Multilingual Named Entity Recognition by XLM-Roberta model with Conditional Random Fields, using zero shot learning. Training with [Pytorch Lightning](https://www.pytorchlightning.ai/) and inference with [ONNX Runtime](https://www.onnxruntime.ai/). The model is finetuned upon [Hugging Face](https://huggingface.co/)'s [xlm-roberta](https://huggingface.co/xlm-roberta-base) models.

## Installation
```
conda create --name mner python=3.8.8
conda activate mner
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
ner = multiner.MultiNerInfer("ner_models/gold_model") # Load pretrained model.
text = """World War II or the Second World War, often abbreviated as WWII or WW2, was a global war that lasted from 1939 to 1945. It involved the vast majority of the world's countries—including all the great powers—forming two opposing military alliances: the Allies and the Axis. In a state of total war, directly involving more than 100 million personnel from more than 30 countries, the major participants threw their entire economic, industrial, and scientific capabilities behind the war effort, blurring the distinction between civilian and military resources. World War II was the deadliest conflict in human history, resulting in 70 to 85 million fatalities, with more civilians than military personnel killed. Tens of millions of people died due to genocides (including the Holocaust), premeditated death from starvation, massacres, and disease. Aircraft played a major role in the conflict, including in strategic bombing of population centres, the development of nuclear weapons, and the only two uses of such in war. """
result = ner(text)
pprint.pprint(result)
```

## Training
[Pytorch Lightning](https://www.pytorchlightning.ai/) is used for training. Place data files into data folder. Then run the following command.
```bash
python train.py --train_path "data/train.txt" --val_path "data/dev.txt" --test_path "data/test.txt" --tags_path "data/tags.txt" --gpus 1
```
For testing the model, run the following command.
```bash
python test.py --test_path "data/test.txt" --tags_path "data/tags.txt" --gpus 1
```

## Pretrained model
[Download](https://drive.google.com/drive/folders/1JMNN9TJWd2oPAl8db1PX-VvXmZMw9h0z?usp=sharing) model (`model.pt`) and tag (`tags.txt`) file from following link. Place it to ![ner_models/gold_model](/ner_models/gold_model) folder. 

## Serving model with Flask
Assuming the model files are at ner_models folder, all you need is to run **app.py**
```bash
python app.py
```

If you want to use [ONNX Runtime](https://www.onnxruntime.ai/) in your Flask app, just add --onnx parameter.
```bash
python app.py --onnx
```

## Onnx Runtime Inference
If you want to use onnx runtime, place torch model into ![ner_models/gold_model](/ner_models/gold_model). Then run the **util_scripts/export2onnx.py** script. Then, you do inference as follows.

```python
import multiner
import pprint
ner = multiner.MultiNerInferONNX("ner_models/gold_model") # Load pretrained model.
text = """World War II or the Second World War, often abbreviated as WWII or WW2, was a global war that lasted from 1939 to 1945. It involved the vast majority of the world's countries—including all the great powers—forming two opposing military alliances: the Allies and the Axis. In a state of total war, directly involving more than 100 million personnel from more than 30 countries, the major participants threw their entire economic, industrial, and scientific capabilities behind the war effort, blurring the distinction between civilian and military resources. World War II was the deadliest conflict in human history, resulting in 70 to 85 million fatalities, with more civilians than military personnel killed. Tens of millions of people died due to genocides (including the Holocaust), premeditated death from starvation, massacres, and disease. Aircraft played a major role in the conflict, including in strategic bombing of population centres, the development of nuclear weapons, and the only two uses of such in war. """
result = ner(text)
pprint.pprint(result)
```

## References
- [XLM-RoBERTa Official Paper](https://arxiv.org/pdf/1911.02116.pdf)
- [XLM-RoBERTa Hugging Face docs](https://huggingface.co/transformers/model_doc/xlmroberta.html)

## Citation

```bibtex
@article{xlmroberta,
	title = {Unsupervised {Cross}-lingual {Representation} {Learning} at {Scale}},
	url = {http://arxiv.org/abs/1911.02116},
	abstract = {This paper shows that pretraining multilingual language models at scale leads to significant performance gains for a wide range of cross-lingual transfer tasks. We train a Transformer-based masked language model on one hundred languages, using more than two terabytes of filtered CommonCrawl data. Our model, dubbed XLM-R, significantly outperforms multilingual BERT (mBERT) on a variety of cross-lingual benchmarks, including +14.6\% average accuracy on XNLI, +13\% average F1 score on MLQA, and +2.4\% F1 score on NER. XLM-R performs particularly well on low-resource languages, improving 15.7\% in XNLI accuracy for Swahili and 11.4\% for Urdu over previous XLM models. We also present a detailed empirical analysis of the key factors that are required to achieve these gains, including the trade-offs between (1) positive transfer and capacity dilution and (2) the performance of high and low resource languages at scale. Finally, we show, for the first time, the possibility of multilingual modeling without sacrificing per-language performance; XLM-R is very competitive with strong monolingual models on the GLUE and XNLI benchmarks. We will make our code, data and models publicly available.},
	urldate = {2021-05-14},
	journal = {arXiv:1911.02116 [cs]},
	author = {Conneau, Alexis and Khandelwal, Kartikay and Goyal, Naman and Chaudhary, Vishrav and Wenzek, Guillaume and Guzmán, Francisco and Grave, Edouard and Ott, Myle and Zettlemoyer, Luke and Stoyanov, Veselin},
	month = apr,
	year = {2020},
	note = {arXiv: 1911.02116},
	keywords = {Computer Science - Computation and Language},
}
```
