import json
import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

DATASET_PATH = os.path.join("data", "data.json")


def load_data(dataset_path):
    with open(dataset_path, 'r') as fp:
        data = json.load(fp)

    # convert lists to numpy arrays
    inputs = np.array(data["mfcc"])
    labels = np.array(data["labels"])

    return inputs, labels


if __name__ == "__main__":
    # load data
    inputs, labels = load_data(DATASET_PATH)

    # split into train and test set
    inputs_train, inputs_test, labels_train, labels_test = train_test_split(inputs,
                                                                            labels,
                                                                            test_size=.3)

    # build the network architecture
    model = tf.keras.Sequential([
        # input layer
        tf.keras.layers.Flatten(input_shape=(inputs.shape[1], inputs.shape[2])),

        # 1st hidden layer
        tf.keras.layers.Dense(512, activation="relu"),

        # 2nd hidden layer
        tf.keras.layers.Dense(256, activation="relu"),

        # 3rd hidden layer
        tf.keras.layers.Dense(64, activation="relu"),

        # output layer
        tf.keras.layers.Dense(10, activation="softmax")  # using 10 because 10 classes
        # softmax normalizes values to sum to 1. highest value is chosen for classifier with percentage
    ])

    # compile network
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer,
                  loss="sparse_categorical_crossentropy",  # since multiclass classifier
                  metrics=["accuracy"])

    model.summary()

    # train network
    model.fit(inputs_train,
              labels_train,
              validation_data=(inputs_test, labels_test),
              epochs=50,
              batch_size=32)
    # batch size -> calculate gradient over batch_size,
    # lower is quick but inaccurate
    # full batch, slow, memory intensive, accurate
    # mini batch (16-128), best of both worlds
