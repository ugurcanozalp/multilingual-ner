import os
import glob
import pickle
from typing import Union,List,Tuple,Dict
import numpy as np
from multiner.utils import CustomTokenizerNP
from multiner.utils import CRFNumpy
import onnxruntime

class MultiNerInferONNX(object):
    
    def __init__(self, model_path:str, roberta_path: str=None, 
        model_name:str="model.onnx", batch_length_limit:int = 380):
        """Inference class for Multilingual Named Entity Recognition model with ONNX
        
        Args:
            model_path (str, required): Path where ONNX model and CRF model is available
            tags_path (str, optional): File path of entity tags list
            roberta_path (str, optional): Path of xlm-roberta model, for tokenizer.
            model_name (str, optional): Name of the onnx model.
            batch_length_limit (int, optional): Number of maximum roberta tokens possible for a single instance.
        """
        self.tags = []
        with open(os.path.join(model_path, "tags.txt")) as f:
            for line in f:
                self.tags.append(line.strip())
        n_labels=len(self.tags)

        roberta_path = "xlm-roberta-base" if roberta_path is None else roberta_path
        self.ort_session = onnxruntime.InferenceSession(os.path.join(model_path, model_name))
        with open(os.path.join(model_path, "crf.pickle"), "rb") as f:
        	crf_np_weights = pickle.load(f)

        self.crf = CRFNumpy(**crf_np_weights)
        self.tokenizer = CustomTokenizerNP(vocab_path=roberta_path, batch_length_limit=batch_length_limit)
        
    def __call__(self, inputs:Union[List[str], str]) -> Union[List[List[Dict]], List[Dict]]:
        """Performs end-to-end prediction, given list of long raw texts.
        
        Args:
            inputs (Union[List[str], str]): List of raw texts /or/ single raw text
        
        Returns:
            Union[List[List[Dict]], List[Dict]]: Found entities as a dictionary.
        """

        results = [] 
        single_input =  isinstance(inputs,str)
        inputs = [inputs] if single_input else inputs
        for input_ in inputs:
            outputs_flat, spans_flat = [], [],
            for batch, spans in self.preprocess(input_):
                ort_inputs = {onnx_input.name: x for onnx_input, x in zip(self.ort_session.get_inputs(), batch)}
                logits, pad_mask = self.ort_session.run(None, ort_inputs)
                print(logits.shape)
                print(pad_mask.shape)
                output = self.crf.viterbi_decode(logits.transpose(1,0,2), pad_mask.transpose(1,0))
                outputs_flat.extend(sum(output, []))
                spans_flat.extend(sum(spans, []))

            outputs = self.postprocess(input_, outputs_flat, spans_flat)
            results.append(outputs)

        return results[0] if single_input and results else results

    def preprocess(self, raw_text:str, batch_size:int=16) -> Tuple[np.ndarray,List[Tuple]]:
        """Preprocess raw text input and split into batches and yield them.
        
        Args:
            raw_text (str): Input text
            batch_size (int, optional): Batch size for Inference. 
        
        Yields:
            Tuple[np.ndarray, List[Tuple]]: Network inputs and word spans.
        """
        for batch, words, spans in self.tokenizer.consume_text(raw_text, batch_size=batch_size):
            yield batch, spans

    def postprocess(self, raw_text:str, output:List[int], span:List[Tuple]) -> List[Dict]:
        """Merges outputs with spans
        
        Args:
            raw_text (str): raw text
            output (List[int]): entity tags for each word (BIO Tagged)
            span (List[Tuple]): start,end locations of each word as tuple
        
        Returns:
            List[Dict]: List of dictionary containing entity information
        """

        entities = []
        tag_pre = 'O'
        for tag_idx,(s,e) in zip(output,span):
            tag = self.tags[tag_idx]
            if tag.startswith('B-'):
                entities.append({'entity': tag[2:], 'start': s, 'end': e})
            elif tag.startswith('I-'):
                if tag_pre == 'O' or tag_pre[2:] != tag[2:]:
                    tag = 'O' # not logical, so assign O to this token.
                else:
                    entities[-1]['end'] = e
            # else: tag is 'O', therefore no more conditions..
            tag_pre = tag

        # add text spans to dictionary
        for entity_ in entities:
            entity_['text'] = raw_text[entity_['start']:entity_['end']]

        return entities

#ner = MultiNerInferenceONNX(model_path="ner_models/gold_model")
#text = """World War II or the Second World War, often abbreviated as WWII or WW2, was a global war that lasted from 1939 to 1945. It involved the vast majority of the world's countries—including all the great powers—forming two opposing military alliances: the Allies and the Axis. In a state of total war, directly involving more than 100 million personnel from more than 30 countries, the major participants threw their entire economic, industrial, and scientific capabilities behind the war effort, blurring the distinction between civilian and military resources. World War II was the deadliest conflict in human history, resulting in 70 to 85 million fatalities, with more civilians than military personnel killed. Tens of millions of people died due to genocides (including the Holocaust), premeditated death from starvation, massacres, and disease. Aircraft played a major role in the conflict, including in strategic bombing of population centres, the development of nuclear weapons, and the only two uses of such in war. """
#result  = ner(text)