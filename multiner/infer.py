import os
import glob
from typing import Union,List,Tuple,Dict
import torch
from multiner.model import XLMRobertaNer
from multiner.utils import CustomTokenizer

class MultiNerInfer(object):

    def __init__(self, model_path:str, roberta_path: str=None, 
        model_name:str="model.pt", batch_length_limit:int = 380):
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
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        roberta_path = "xlm-roberta-base" if roberta_path is None else roberta_path
        self.model = XLMRobertaNer(n_labels=n_labels, roberta_path=roberta_path).to(self.device)
        state_dict = torch.load(os.path.join(model_path, model_name))
        self.model.load_state_dict(state_dict, strict=False)
        self.model.eval()
        self.tokenizer = CustomTokenizer(vocab_path=roberta_path, to_device=self.device)

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
                output, pad_mask = self.model.predict(batch)
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
