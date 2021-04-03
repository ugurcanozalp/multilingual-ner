import os
import glob
from typing import Union,List,Tuple,Dict
import torch
from multiner.model import MultiNer
from multiner.utils import CustomTokenizer

class MultiNerInference(object):
    
    def __init__(self, roberta_path:str=None, tags_path:str=None, 
            batch_length_limit:int = 380, load_backbone:bool=False, half_precision:bool=False, device:str=None):
        """Inference class for Multilingual Named Entity Recognition model 
        
        Args:
            roberta_path (str, optional): Path where XLM Roberta model files are available (weight file is not required, used for tokenizer)
            tags_path (str, optional): File path of entity tags list
            batch_length_limit (int, optional): Number of maximum roberta tokens possible for a single instance.
            load_backbone (bool, optional): Switch of loading pretrained xlm-roberta-base weigts to backbone.
            half_precision (bool, optional): Switch of half precision usage (approximately halves occupied memory)
        """
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else cpu
        else:
            self.device = device 

        self.tags = []
        if tags_path is not None:
            with open(tags_path) as f:
                for line in f:
                    self.tags.append(line.strip())
        
        n_labels = 37 if not self.tags else len(self.tags)

        roberta_path = "xlm-roberta-base" if roberta_path is None else roberta_path
        if half_precision:
            self.model = MultiNer(n_labels=n_labels, roberta_path=roberta_path, load_backbone=load_backbone).half().to(self.device)
        else:
            self.model = MultiNer(n_labels=n_labels, roberta_path=roberta_path, load_backbone=load_backbone).to(self.device)
            
        self.tokenizer = CustomTokenizer(vocab_path=roberta_path, to_device=self.device)
        
        self.initialized = False

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

    @classmethod
    def load_model(cls, model_path:str, roberta_path:str=None, batch_length_limit:int=380, device=None):
        """Load pretrained model.
        
        Args:
            model_path (str): Folder path of pretrained ner model (tags.txt file and model.pt file should be available) 
            roberta_path (str, optional): Folder path of roberta utilities (if not given, default will be used from huggingface)
            version (int, optional): version of the model
            batch_length_limit (int, optional): Number of maximum roberta tokens possible for a single instance.
        
        Returns:
            infer: Inference module for Named Entity Recognition
        """
        tags_path = os.path.join(model_path,'tags.txt')
        infer = cls(roberta_path, tags_path, batch_length_limit, load_backbone=False, device=device)       
        state_dict = torch.load(os.path.join(model_path,"model.pt"))
        infer.model.load_state_dict(state_dict, strict=False)
        infer.model.eval()
        return infer

    # method for torchserve
    def initialize(self, context):
        #  load the model
        self.manifest = context.manifest
        properties = context.system_properties
        model_dir = properties.get("model_dir")
        tags_path = os.path.join(model_dir, "tags.txt")
        roberta_path = model_dir
        self.device = torch.device("cuda:" + str(properties.get("gpu_id")) if torch.cuda.is_available() else "cpu")
        self.tokenizer = CustomTokenizer(vocab_path=roberta_path, to_device=self.device)
        self.tags = []
        with open(tags_path) as f:
            for line in f:
                self.tags.append(line.strip())

        #self.model = MultiNer(n_labels=len(self.tags), roberta_path=roberta_path, load_backbone=False).to(self.device)  
        state_dict = torch.load(os.path.join(model_dir,"model.pt"))
        self.model.load_state_dict(state_dict, strict=False)
        self.model.eval()
        self.initialized = True

    def handle(self, data, context):
        if not self.initialized:
            self.initialize(context)

        text = data[0].get("data")
        if text is None:
            text = data[0].get("body")

        text = text.decode('utf-8')

        return [self(text)]

#torch-model-archiver --model-name multiner --version 1.0 --model-file multiner/model/model.py --serialized-file ner_models/gold/model.pt --export-path model_store/ --extra-files "ner_models/gold/tags.txt,ner_models/xlm-roberta-base/config.json,ner_models/xlm-roberta-base/tokenizer.json,ner_models/xlm-roberta-base/sentencepiece.bpe.model,multiner.zip" --handler multiner/infer.py
#torchserve --start --ncs --model-store model_store/ --models multiner.mar