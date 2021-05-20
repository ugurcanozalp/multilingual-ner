from transformers import XLMRobertaTokenizerFast
from typing import Union,List,Tuple
import numpy as np
from typing import List,Dict,Tuple, Callable
from nltk.tokenize import RegexpTokenizer
from multiner.utils.abbreviations import abbreviations
import nltk

def stack_with_padding(array_list, padding_value, pad_to=None):
    max_seq_len = max([array.shape[0] for array in array_list])
    pad_to = max_seq_len if pad_to is None or pad_to<max_seq_len else pad_to
    new_array_list = [np.concatenate((array, padding_value*np.zeros(pad_to-array.shape[0], dtype=array.dtype))) for array in array_list]
    return np.stack(new_array_list)
    
class CustomTokenizerNP(object):
    MAX_LEN=512
    def __init__(self, vocab_path:str, do_lower_case:bool=False, batch_length_limit:int=380, to_device:str='cpu'):
        """Generic Tokenizer for XLM Roberta models.
        
        Args:
            vocab_path (str): Path of tokenizer files
            do_lower_case (bool, optional): Lowercase switch before tokenization for roberta tokenizer
            batch_length_limit (int, optional): Maximum allowed number of roberta token for a single batch instance
            to_device (str, optional): Device for output tensors
        """
        super(CustomTokenizerNP,self).__init__()
        self.roberta_tokenizer = XLMRobertaTokenizerFast.from_pretrained(vocab_path, do_lower_case=do_lower_case)
        self.batch_length_limit = batch_length_limit
        self.to_device = to_device
        self.sent_tokenizer = nltk.data.load("tokenizers/punkt/{0}.pickle".format('turkish'))
        self.sent_tokenizer._params.abbrev_types.update(abbreviations)
        pattern = r'''(?x)          # set flag to allow verbose regexps
                (?:[A-Z]\.)+        # abbreviations, e.g. U.S.A. or U.S.A # 
              | (?:\d+\.)           # numbers
              | \w+(?:[-.]\w+)*     # words with optional internal hyphens
              | \$?\d+(?:.\d+)?%?   # currency and percentages, e.g. $12.40, 82%
              | \.\.\.              # ellipsis, and special chars below, includes ], [
              | [-\]\[.,;"'?():_`“”/°º‘’″…#$%()*+<>=@\\^_{}|~❑&§]
            '''
        # #$%()*+<>=@\\^_{|~❑&
        self.word_tokenizer = RegexpTokenizer(pattern)

    def tokenize_words_batch(self, inputs: Union[List[str], List[List[str]]], pad_to: int = None) -> np.ndarray:
        """Roberta tokenization of given list of words as batch.
        
        Args:
            inputs (Union[List[str], List[List[str]]]): Batch of list of words to be tokenized.
            pad_to (int, optional): Padding size of batch
        
        Returns:
            np.ndarray: Roberta model inputs
        """
        if not isinstance(inputs,List):
            print(inputs)
        assert isinstance(inputs,List),"input must be list"

        single_input = isinstance(inputs[0], str)
        batch = [inputs] if single_input else inputs

        input_ids, attention_mask, token_type_ids, mask = [], [], [], []
        for tokens in batch:
            input_ids_tmp, attention_mask_tmp, token_type_ids_tmp, mask_tmp = self._tokenize_words(tokens)
            input_ids.append(input_ids_tmp)
            attention_mask.append(attention_mask_tmp)
            token_type_ids.append(token_type_ids_tmp)
            mask.append(mask_tmp)

        input_ids = stack_with_padding(input_ids, padding_value=self.roberta_tokenizer.pad_token_id, pad_to=pad_to)
        attention_mask = stack_with_padding(attention_mask, padding_value=0, pad_to=pad_to)
        token_type_ids = stack_with_padding(token_type_ids, padding_value=0, pad_to=pad_to)
        mask = stack_with_padding(mask, padding_value=0, pad_to=pad_to)

        if input_ids.shape[-1]>self.MAX_LEN:
            input_ids = input_ids[:,:,:self.MAX_LEN]
            attention_mask = attention_mask[:,:,:self.MAX_LEN]
            token_type_ids = token_type_ids[:,:,:self.MAX_LEN]
            mask = mask[:,:,:self.MAX_LEN]
        elif pad_to is not None and pad_to<batched_tokens.shape[1]:
            input_ids = input_ids[:,:,:self.pad_to]
            attention_mask = attention_mask[:,:,:self.pad_to]
            token_type_ids = token_type_ids[:,:,:self.pad_to]
            mask = mask[:,:,:self.pad_to]

        if single_input:
            return input_ids[0], attention_mask[0], token_type_ids[0], mask[0]
        else:
            return input_ids, attention_mask, token_type_ids, mask

    def _tokenize_words(self, words:List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Tokenize given list of words with XLM Roberta Tokenizer.

        Args:
            words (List[str]): List of words

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Model inputs
        """
        subtokenized = []
        mask = []
        for word in words:
            subtokens = self.roberta_tokenizer.tokenize(word)
            subtokenized+=subtokens
            n_subtoken = len(subtokens)
            if n_subtoken>=1:
                mask = mask + [1] + [0]*(n_subtoken-1)

        subtokenized = [self.roberta_tokenizer.cls_token] + subtokenized + [self.roberta_tokenizer.sep_token]
        mask = [0] + mask + [0]

        input_ids = np.array(self.roberta_tokenizer.convert_tokens_to_ids(subtokenized), dtype=np.int)
        attention_mask = np.ones(len(mask), dtype=np.int)
        token_type_ids = np.zeros(len(mask), dtype=np.int)
        mask = np.array(mask, dtype=np.int)

        return input_ids, attention_mask, token_type_ids, mask

    def yield_words(self, raw_text:str) -> Tuple[List[str], List[Tuple]]:
        """Splits given raw text to sentences
        
        Args:
            raw_text (str): raw text
        
        Yields:
            Tuple[List[str], List[Tuple]]: List of words and list of spans
        """
        for offset, ending in self.sent_tokenizer.span_tokenize(raw_text):
            sub_text = raw_text[offset:ending]
            words, spans = [], []
            flush = False
            total_subtoken = 0
            for s,e in self.word_tokenizer.span_tokenize(sub_text):
                flush = True
                s += offset
                e += offset
                words.append(raw_text[s:e])
                spans.append((s,e))
                total_subtoken += len(self.roberta_tokenizer.tokenize(words[-1]))
                if (total_subtoken > self.batch_length_limit): 
                    yield words[:-1],spans[:-1]
                    spans = spans[len(spans)-1:]
                    words = words[len(words)-1:]
                    total_subtoken = sum([len(self.roberta_tokenizer.tokenize(word)) for word in words])
                    flush = False

            if flush and len(spans) > 0:
                yield words,spans

    def consume_text(self, raw_text, batch_size=16):
        """Consume given raw text into small pieces for neural network model with fixed batch size
        
        Args:
            raw_text (str): raw text
            batch_size (int, optional): batch size for consuming text
        
        Yields:
            Tuple[
                Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], 
                List[List[str]],
                List[List[Tuple[int]]]]: Model inputs as tuple, lists of words for the batch, lists of spans for the batch
        """
        batch_words, batch_spans = [], []
        flag = False
        for words, spans in self.yield_words(raw_text):
            flag = True
            batch_words.append(words)
            batch_spans.append(spans)
            if len(batch_spans) >= batch_size:
                input_ids, attention_mask, token_type_ids, mask = self.tokenize_words_batch(batch_words)
                yield (input_ids, attention_mask, token_type_ids, mask), batch_words, batch_spans
                batch_words, batch_spans = [], []
        if flag and len(batch_words) > 0:
            input_ids, attention_mask, token_type_ids, mask = self.tokenize_words_batch(batch_words)
            yield (input_ids, attention_mask, token_type_ids, mask), batch_words, batch_spans


