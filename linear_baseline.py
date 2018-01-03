from sklearn.linear_model import LogisticRegression
import numpy as np

from config import (data_test_file,
                    label_test_file,
                    data_train_file,
                    label_train_file,
)

print("Load test data ...")
X_train, y_train = (np.load(data_train_file), np.load(label_train_file))
X_train = X_train.reshape((X_train.shape[0], -1))
X_test, y_test = (np.load(data_test_file), np.load(label_test_file))
X_test = X_test.reshape((X_test.shape[0], -1))

print("Train model ...")
model = LogisticRegression(n_jobs=8)
model.fit(X_train, y_train)

print("Test model ...")
acc = model.score(X_test, y_test)
print("Test accuracy: {:.2f}".format(acc))
