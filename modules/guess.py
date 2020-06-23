import numpy as np
from transformers import BertTokenizer

from string import punctuation
from nltk.corpus import stopwords

from typing import List, Tuple, Union

punctuation += '«»—…“”*№–'
punctuation = set(punctuation)
stopwords = set(stopwords.words('russian')) | set(stopwords.words('english'))


class Guess:
    """
    Этот класс используется для структурированного хранения предсказаний модели для маскированных токенов.
    """

    def __init__(self,
                 words: List[str],
                 probs: Union[List[float], np.ndarray]) -> None:
        self.words = words
        self.probs = np.array(probs) if isinstance(probs, list) else probs

    @classmethod
    def from_model_predictions(cls,
                               probs: np.ndarray,
                               tokenizer: BertTokenizer,
                               max_len: int = 1000):
        probs_with_indices = list(enumerate(probs))
        best_probs_with_indices = sorted(probs_with_indices, key=lambda x: x[1], reverse=True)

        best_words = {}
        for i, prob in best_probs_with_indices:
            word = tokenizer.convert_ids_to_tokens(i).lower()
            if word in punctuation | stopwords or word in best_words:
                continue

            best_words[word] = prob

            if len(best_words) == max_len:
                break

        best_probs = list(best_words.values())
        best_words = list(best_words.keys())

        return cls(best_words, best_probs)

    def get_best_guess(self) -> Union[Tuple[str, float], None]:
        return self.words[0], self.probs[0] if bool(self) else None

    def get_n_best_guesses(self, n: int = 5) -> Union[List[str], None]:
        return self.words[:n] if bool(self) else None

    def __bool__(self) -> bool:
        return bool(self.words) and self.probs.shape[0] == len(self.words)

    def __str__(self) -> str:
        return '\n'.join([f'{word}: {prob:.2f}' for word, prob in zip(self.words, self.probs)])

    def __len__(self) -> int:
        return self.probs.shape[0]

    def __add__(self, x):
        if not self and not x:
            return Guess([], [])
        if not self:
            return x
        if not x:
            return self

        combined_words = set(self.words) & set(x.words)
        if not combined_words:
            return Guess([], [])

        new_words = {}

        i, j = 0, 0
        while i < len(self) or j < len(x):
            if j == len(x) or (i < len(self) and self.probs[i] > x.probs[j]):
                if self.words[i] not in combined_words:
                    i += 1
                    continue
                new_words[self.words[i]] = (new_words.get(self.words[i], self.probs[i]) + self.probs[i]) / 2
                i += 1
            else:
                if x.words[j] not in combined_words:
                    j += 1
                    continue
                new_words[x.words[j]] = (new_words.get(x.words[j], x.probs[j]) + x.probs[j]) / 2
                j += 1

        new_len = min(len(self), len(x))
        sorted_items = sorted(new_words.items(), key=lambda x: x[1], reverse=True)[:new_len]
        new_words = [x[0] for x in sorted_items]
        new_probs = np.array([x[1] for x in sorted_items])

        return Guess(new_words, new_probs)
