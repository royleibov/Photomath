import numpy as np
import matplotlib.pyplot as plt
import pickle
import gzip
import random
import json
import tensorflow as tf
import pandas as pd

(x_train, y_train), (test_x, test_y) = tf.keras.datasets.mnist.load_data()

train_x = x_train[:50000:].reshape((50000, 28, 28, 1)) / 255.0
train_y = y_train[:50000:]
valid_x = x_train[50000:].reshape((10000, 28, 28, 1)) / 255.0
valid_y = y_train[50000:]
test_x = test_x.reshape((10000, 28, 28, 1)) / 255.0

CNN = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (5, 5), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(64, (5, 5), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(100, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

CNN.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

CNN.fit(train_x, train_y, validation_data=(valid_x, valid_y), batch_size=10, epochs=5)
