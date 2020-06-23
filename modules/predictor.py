import torch
from scipy.special import softmax
from transformers import BertTokenizer, BertForMaskedLM

from overrides import overrides
from abc import ABC, abstractmethod

from typing import List, Tuple, Union

from .guess import Guess


class MLMPredictor(ABC):
    """
    Это абстрактный класс для предиктора MLM-модели.
    """

    def __init__(self,
                 bert_path: str = 'rubert_cased_L-12_H-768_A-12_pt',
                 device: str = 'cpu') -> None:
        self.tokenizer = BertTokenizer.from_pretrained(bert_path, do_lower_case=False)
        self.mask_token = self.tokenizer.mask_token

        self.model = BertForMaskedLM.from_pretrained(bert_path)
        self.model.eval()

        self.device = torch.device(device)
        self.model.to(self.device)

    @abstractmethod
    def tokenize(*args, **kwargs):
        pass

    @abstractmethod
    def predict(*args, **kwargs):
        pass


class BERTMLMPredictor(MLMPredictor):
    """
    Этот класс позволяет получить предсказания MLM-модели для маскированного токена.
    """

    @overrides
    def tokenize(self, text: str) -> Tuple[List[str], int]:
        tokens = self.tokenizer.tokenize(text.strip(), add_special_tokens=True)

        mask_indices = [i for i, token in enumerate(tokens) if token == self.mask_token]
        if len(mask_indices) == 0:
            raise IOError(f'The mask token is not found in model input: {text}.')
        if len(mask_indices) > 1:
            raise IOError(f'The mask token occurs more than once in model input: {text}.')
        mask_idx = mask_indices[0]

        return tokens, mask_idx

    @overrides
    def predict(self,
                text: Union[str, List[str]]) -> Guess:
        tokens, mask_idx = self.tokenize(text) if isinstance(text, str) \
            else self.tokenize(' '.join(text))

        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        with torch.no_grad():
            model_output = self.model(torch.tensor([token_ids]).to(self.device))[0][0]

        masked_probs = softmax(model_output[mask_idx].cpu().numpy())
        guess = Guess.from_model_predictions(masked_probs, self.tokenizer)

        return guess
