#!/usr/bin/env python3

import urllib.request
import gzip
import shutil
import os
import os.path

from config import base_path, word2vec_file, word2vec_url

def download_and_unzip(url, dst):
    print("Safe unzipped content of {} as {} ...".format(url, dst))
    with urllib.request.urlopen(url) as response:
        with gzip.GzipFile(fileobj=response) as uncompressed:
            with open(dst, 'wb') as f:
                shutil.copyfileobj(uncompressed, f)


def main():
    if os.path.isdir(base_path):
        print('Data directory file found.')
    else:
        print('Make data directory {}'.format(base_path))
        os.mkdir(base_path)

    if os.path.isfile(word2vec_file):
        print('Word2vec binary found.')
    else:
        download_and_unzip(word2vec_url, word2vec_file)


if __name__ == '__main__':
    main()
