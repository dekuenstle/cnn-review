#!/usr/bin/env python3

import os

import numpy as np

from vectorize import load_word2vec, word2vec_sentences
from config import (random_seed,
                    polarity_dataset_dir,
                    valid_ratio,
                    test_ratio,
                    max_word_num,
                    word2vec_dim,
                    data_train_file,
                    label_train_file,
                    data_valid_file,
                    label_valid_file,
                    data_test_file,
                    label_test_file)


def read_files(dir):
    files = os.listdir(dir)
    content = ""
    for file_name in sorted(files):
        path = os.path.join(dir, file_name)
        with open(path, 'r') as f:
            content += f.read()
    return content


def read_dataset(shuffled):
    print("Read dataset from {} ...".format(polarity_dataset_dir))
    polarity_pos_dir = os.path.join(polarity_dataset_dir, 'pos')
    polarity_neg_dir = os.path.join(polarity_dataset_dir, 'neg')
    pos_content = read_files(polarity_pos_dir).splitlines()
    neg_content = read_files(polarity_neg_dir).splitlines()

    content = np.array(pos_content + neg_content)
    n = len(content)
    labels = np.empty((n, ))
    labels[:len(pos_content)] = 1
    labels[len(pos_content):] = 0

    if shuffled:
        print("Shuffle data and labels ...".format())
        indices = np.arange(n)
        np.random.shuffle(indices)
        return content[indices], labels[indices]
    else:
        return content, labels



def save_splitted(data, label):
    n = len(data)
    n_valid = int(valid_ratio * n)
    n_test = int(test_ratio * n)
    print("Split into training, validation ({:.0f}%) and test ({:.0f}%) set ..."
          .format(100 * valid_ratio, 100 * test_ratio))

    def save_numpy(file, arr):
        print("  Save array {} to {} ...".format(arr.shape, file))
        np.save(file, arr)
    save_numpy(data_train_file, data[(n_test + n_valid):])
    save_numpy(label_train_file, label[(n_test + n_valid):])
    save_numpy(data_valid_file, data[n_test:(n_test + n_valid)])
    save_numpy(label_valid_file, label[n_test:(n_test + n_valid)])
    save_numpy(data_test_file, data[:n_test])
    save_numpy(label_test_file, label[:n_test])


def main():
    np.random.seed(random_seed)
    sentences, labels = read_dataset(shuffled=True)
    data = word2vec_sentences(sentences, print_stat=True)
    save_splitted(data, labels)


if __name__ == "__main__":
    main()
