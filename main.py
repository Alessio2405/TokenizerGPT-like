from transformers import AutoTokenizer
from collections import defaultdict
from utils import tokenize_corpus, create_alphabet, split_words 


def main():
    corpus = [
        "This is a test tokenizer.",
        "Tokenization is fundamental in NLP.",
        "There are several tokenizer algorithms."
    ]

    word_freqs = tokenize_corpus(corpus, "gpt2")
    print(word_freqs)

    full_alphabet = create_alphabet(word_freqs)
    print(full_alphabet)

    splits = split_words(word_freqs)
    print(splits)

if __name__ == "__main__":
    main()