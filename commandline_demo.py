#!/usr/bin/env python3

from keras.models import load_model
import numpy as np

from vectorize import load_word2vec, word2vec_sentences
from config import (model_file,
                    minibatch_size
)

print("Load model {} ...".format(model_file))
model = load_model(model_file)
wv = load_word2vec()

print("\n\n\n\n\n\n")
while True:
    sentence = input("How was the film? ").lower()
    data = word2vec_sentences([sentence], wv, print_stat=False)
    print("Predict ...")
    label = model.predict(data)[0, 0]

    if label > 0.33 and label < 0.67:
        print("==> I'm not sure what you mean ... ({:.0f}% pos)".format(label * 100))
    else:
        if label < 0.5:
            label_desc = "do not like"
        else:
            label_desc = "like"
        print("==> You {} it! ({:.0f}% pos)".format(label_desc, label * 100))
    print()
