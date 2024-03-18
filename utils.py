from transformers import AutoTokenizer
from collections import defaultdict

def tokenize_corpus(corpus, tokenizer_model):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)
    word_freqs = defaultdict(int)

    for text in corpus:
        words_with_offsets = tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(text)
        new_words = [word for word, offset in words_with_offsets]
        for word in new_words:
            word_freqs[word] += 1

    return word_freqs

def create_alphabet(word_freqs):
    full_alphabet = []
    for word in word_freqs.keys():
        for letter in word:
            if letter not in full_alphabet:
                full_alphabet.append(letter)
    return full_alphabet

def split_words(word_freqs):
    splits = {word: [c for c in word] for word in word_freqs.keys()}
    return splits