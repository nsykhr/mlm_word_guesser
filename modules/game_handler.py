from .guess import Guess
from .predictor import BERTMLMPredictor


class GameHandler:
    """
    Этот класс отвечает за взаимодействие с пользователем и координирует работу остальных классов.
    """

    def __init__(self,
                 bert_path: str = 'rubert_cased_L-12_H-768_A-12_pt',
                 max_rounds: int = 10,
                 prob_threshold: float = 0.5,
                 device: str = 'cpu') -> None:
        self.predictor = BERTMLMPredictor(bert_path, device)
        self.guess = Guess([], [])
        self.max_rounds = max_rounds
        self.prob_threshold = prob_threshold

    def __update_guess(self, text: str) -> None:
        new_guess = self.predictor.predict(text)
        self.guess += new_guess

    def __play_round(self) -> None:
        text = input()
        self.__update_guess(text)

    def play_game(self, n: int = 5) -> None:
        for _ in range(self.max_rounds):
            self.__play_round()

            if not self.guess:
                print(f'Вы меня запутали, я сдаюсь.')
                return

            best_word, best_prob = self.guess.get_best_guess()
            if best_prob > self.prob_threshold or len(self.guess) == 1:
                print(f'Я думаю, вы загадали слово "{best_word}".')
                return

        best_guesses = self.guess.get_n_best_guesses(n)
        if len(best_guesses) == 1:
            print(f'Я думаю, вы загадали слово "{best_guesses[0]}".')
            return

        best_guesses = list(map(lambda x: '"' + x + '"', best_guesses))

        print(f'Вот мои лучшие догадки: {", ".join(best_guesses)}.')
