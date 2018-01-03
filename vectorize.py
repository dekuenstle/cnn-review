#!/usr/bin/env python3

from gensim.models import KeyedVectors
from gensim.utils import tokenize
import numpy as np

from config import random_seed, word2vec_file, word2vec_dim, max_word_num

def load_word2vec():
    print("Load Word2Vec from {} ...".format(word2vec_file))
    return KeyedVectors.load_word2vec_format(word2vec_file,
                                             binary=True)

def word2vec_unknown():
    np.random.seed(random_seed)
    return np.random.uniform(-1, 1, (word2vec_dim,))

def word2vec_sentence(sentence, wv, arr=None, zero_padding=True, unknown_vec=None):
    tokens = list(tokenize(sentence, lowercase=False, deacc=False))
    if arr is None:
        n_used_token = len(tokens)
        arr = np.empty((n_used_token, word2vec_dim))
    else:
        n_used_token = min(arr.shape[0], len(tokens))
    if unknown_vec is None:
        unknown_vec = word2vec_unknown()

    skipped_ind = []
    for i, token in enumerate(tokens[:n_used_token]):
        if token in wv:
            arr[i,:] = wv[token]
        else:
            skipped_ind.append(i)
            arr[i,:] = unknown_vec
    if zero_padding and n_used_token < arr.shape[0]:
        arr[n_used_token:,:] = 0

    return arr, tokens, skipped_ind


def word2vec_sentences(sentences, wv=None, print_stat=True):
    if wv is None:
        wv = load_word2vec()
    data = np.empty((len(sentences), max_word_num, word2vec_dim))
    n = len(sentences)
    unknown_count = dict()
    token_set = set()
    n_unknown = np.empty((n,))
    n_token = np.empty((n,))
    offset = np.empty((n,))

    print("Transform words to vectors ...", end='')
    for i in range(n):
        if (i + 1) % 5000 == 0:
            print(".", end="")
        sentence_arr, tokens, unknown_ind = word2vec_sentence(sentences[i], wv, data[i])
        n_unknown[i] = len(unknown_ind)
        n_token[i] = len(tokens)
        token_set.update(tokens)
        for ind in unknown_ind:
            unknown_tok = tokens[ind]
            if unknown_tok in unknown_count:
                unknown_count[unknown_tok] += 1
            else:
                unknown_count[unknown_tok] = 1
    print(".")

    if print_stat:
        def print_stat_func(arr, sum_n_token, desc):
            print("  {} of {} tokens are {} ({:.1f}%), min {}, max {}, mean {:.2f}, median {}"
                  .format(int(arr.sum()), int(sum_n_token), desc, 100*(arr.sum()/sum_n_token), int(arr.min()),
                          int(arr.max()), arr.mean(), int(np.percentile(arr, 50))))

        print("Print statistics ...")
        n_padded = (max_word_num - n_token).clip(0)
        n_clipped = (n_token - max_word_num).clip(0)
        sum_n_token = n_token.sum()
        print("  Dataset contains {} sentences, fixed sentence length is {}, number of unique tokens is {}"
              .format(n, max_word_num, len(token_set)))
        print_stat_func(n_token, sum_n_token, "in dataset sentences")
        print_stat_func(n_clipped, sum_n_token, "clipped")
        print_stat_func(n_padded, max_word_num * n, "padded")
        print_stat_func(n_unknown, sum_n_token, "unknown")
        common_unknowns = sorted(unknown_count.items(), key=lambda x: x[1])[::-1][:10]
        print("  Most common unknowns: {}"
              .format(", ".join(["{} ({})".format(t, c) for t, c in common_unknowns])))
    return data


def main():
    wv = load_word2vec()
    np.random.seed(random_seed)
    sentence = 'The police in Hintertupfingen is slow today!'
    mat, tok, skipped = word2vec_sentence(sentence, wv)
    print("Full sentence:      ", sentence)
    print("All tokens:         ", tok)
    print("Skipped (zero vec): ", [tok[i] for i in skipped])
    print("Matrix (only 5D):\n", mat[:, :5])

    print()
    mat = word2vec_sentences([sentence], wv)
    print("Matrix (only 10x5D):\n", mat[0, :10, :5])



if __name__ == "__main__":
    main()
