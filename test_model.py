#!/usr/bin/env python3

from keras.models import load_model
import numpy as np

from config import (data_test_file,
                    label_test_file,
                    model_file,
                    minibatch_size
)

print("Load test data ...")
test_data = (np.load(data_test_file), np.load(label_test_file))

print("Load model ...")
model = load_model(model_file)

print("Evaluate ...")
score, acc = model.evaluate(test_data[0], test_data[1], minibatch_size)
print("==> Test accuracy {:.3f}".format(acc))
