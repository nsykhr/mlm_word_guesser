# MLM Word Guesser

This repository is meant to showcase my Object-oriented programming skills in Python and contains a funny little game. The rules are as follows:
1. The player must feed sentences containing a [MASK] token (only one) to the model that was trained on a Masked Language Modeling objective (I used DeepPavlov's `RuBERT`).
2. The model tries to guess the word, narrowing the search at every iteration.
3. As soon as the specified maximum number of rounds was played, the model prints its best guesses.
4. If the model can identify the winner with good certainty (either through the process of elimination or by assigning it a high probability score) before the maximum number of rounds is reached, it prints its guess and ends the game.
5. Finally, if the process of elimination leaves no candidates to choose from, the system accepts its defeat.

To try it, run the `run.py` script or the `MLMWordGuesser` notebook, and don't forget to specify the path to a model that can be loaded with the `transformers` library.

P. S. It is only sensible to mask words that are transformed into one BPE-token with the model's tokenizer. The current version also does not use any lemmatization methods because they can be unreliable and are language-specific.
