import os
import glob
from typing import Union,List,Tuple,Dict
import torch
from multiner.utils import CustomTokenizer
from multiner.model.crf import CRF
import onnxruntime

class MultiNerInferONNX(object):
    
    def __init__(self, model_path:str, roberta_path: str=None, 
        model_name:str="model-optimized.onnx", batch_length_limit:int = 380):
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
        self.crf = CRF(n_labels, batch_first=True)
        self.crf.load_state_dict(torch.load(os.path.join(model_path, "crf_dict.pt")))
        self.tokenizer = CustomTokenizer(vocab_path=roberta_path)
        
    @torch.no_grad()
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
                ort_inputs = {onnx_input.name: self.to_numpy(x) for onnx_input, x in zip(self.ort_session.get_inputs(), batch)}
                logits_np, pad_mask_np = self.ort_session.run(None, ort_inputs)
                logits, pad_mask = self.to_torch(logits_np), self.to_torch(pad_mask_np)
                output = self.crf.decode(logits, pad_mask)
                outputs_flat.extend(output[pad_mask.bool()].reshape(-1).tolist())
                spans_flat += sum(spans, [])

            outputs = self.postprocess(input_, outputs_flat, spans_flat)
            results.append(outputs)

        return results[0] if single_input and results else results

    def preprocess(self, raw_text:str, batch_size:int=16) -> Tuple[torch.Tensor,List[Tuple]]:
        """Preprocess raw text input and split into batches and yield them.
        
        Args:
            raw_text (str): Input text
            batch_size (int, optional): Batch size for Inference. 
        
        Yields:
            Tuple[torch.Tensor, List[Tuple]]: Network inputs and word spans.
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

    @staticmethod
    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    @staticmethod
    def to_torch(array):
        return torch.from_numpy(array)

#ner = MultiNerInferenceONNX(model_path="ner_models/gold_model")
#text = """World War II or the Second World War, often abbreviated as WWII or WW2, was a global war that lasted from 1939 to 1945. It involved the vast majority of the world's countries—including all the great powers—forming two opposing military alliances: the Allies and the Axis. In a state of total war, directly involving more than 100 million personnel from more than 30 countries, the major participants threw their entire economic, industrial, and scientific capabilities behind the war effort, blurring the distinction between civilian and military resources. World War II was the deadliest conflict in human history, resulting in 70 to 85 million fatalities, with more civilians than military personnel killed. Tens of millions of people died due to genocides (including the Holocaust), premeditated death from starvation, massacres, and disease. Aircraft played a major role in the conflict, including in strategic bombing of population centres, the development of nuclear weapons, and the only two uses of such in war. """
#result  = ner(text)