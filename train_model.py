#!/usr/bin/env python3

import numpy as np
import keras
from keras import regularizers
from keras.models import Model
from keras.optimizers import Adam
from keras.losses import binary_crossentropy
from keras.layers import Input, Concatenate, Conv2D, MaxPooling2D, Flatten, Dense, Reshape, Dropout

from config import (data_train_file,
                    label_train_file,
                    data_valid_file,
                    label_valid_file,
                    model_file,
                    minibatch_size,
                    max_word_num,
                    word2vec_dim
)

in_shape = (max_word_num, word2vec_dim)

print("Load training and validation data ...")
train_data = (np.load(data_train_file), np.load(label_train_file))
valid_data = (np.load(data_valid_file), np.load(label_valid_file))

print("Build model ...")
l2_rate = 0.01
n_epochs = 20

in_layer = Input(in_shape)
reshape_layer = Reshape(in_shape + (1, ))(in_layer)
convmax_layers = []
for i in [2, 3, 4]:
    conv_layer = Conv2D(128, kernel_size=(i, in_shape[1]),
                        padding='valid', activation='relu')(reshape_layer)
    pool_layer = MaxPooling2D(pool_size=(in_shape[0] - i + 1, 1))(conv_layer)
    convmax_layers.append(pool_layer)
cat_layer = Concatenate(3)(convmax_layers)
flat_layer = Flatten()(cat_layer)
drop_layer = Dropout(0.5)(flat_layer)
out_layer = Dense(1, activation='sigmoid',
                  kernel_regularizer=regularizers.l2(l2_rate),
                  bias_regularizer=regularizers.l2(l2_rate))(drop_layer)

model = Model(inputs=in_layer, outputs=out_layer)
model.summary()

print("Compile and train model ...")
model.compile(loss=binary_crossentropy,
              optimizer=Adam(lr=0.005),
              metrics=['accuracy'])

model.fit(*train_data,
          batch_size=minibatch_size,
          epochs=n_epochs,
          verbose=1,
          validation_data=valid_data)

print("Save model to {} ...".format(model_file))
model.save(model_file)
