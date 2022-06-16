import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import time

flower_str_list = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']

CURRENT_PATH = os.getcwd()
DATASET_PATH = os.path.join(CURRENT_PATH, 'drive',
                            'MyDrive', 'flower_dataset')
TEST_DATASET_PATH = os.path.join(CURRENT_PATH, 'drive',
                            'MyDrive', 'flower_test_dataset')

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    DATASET_PATH,
    image_size=(256, 256),
    batch_size=64,
    subset='training',
    validation_split=0.2,
    shuffle=True,
    seed=8282
)
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    DATASET_PATH,
    image_size=(256, 256),
    batch_size=64,
    subset='validation',
    validation_split=0.2,
    shuffle=True,
    seed=8282
)
test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    TEST_DATASET_PATH,
    image_size=(256, 256),
    batch_size=64,
    shuffle=True,
    seed=8282
)


def get_overfitting_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(64, (3, 3), padding="same",
                               activation="relu", input_shape=(256, 256, 3)),
        tf.keras.layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(5, activation="softmax")
    ])
    return model

def get_not_alexnet_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), padding="same",
                               activation="relu", input_shape=(256, 256, 3)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(5, activation="softmax")
    ])
    return model


def get_alexnet_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(64, (3, 3), padding="same",
                               activation="relu", input_shape=(256, 256, 3)),
        tf.keras.layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(5, activation="softmax")
    ])
    return model

def get_callback(mode):
    save_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath='flower/checkpoint/' + mode,
        monitor='val_acc',
        mode='max',
        save_weights_only=True,
        save_freq='epoch'
    )
    return save_callback


def simulation(mode, epochs, model):
    model_path = os.path.join(os.getcwd(), 'drive',
                              'MyDrive', 'flower_model', mode)
    logs_path = os.path.join(os.getcwd(), 'drive', 'MyDrive',
                             'flower_logs', mode)
    model.summary()
    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer="adam", metrics=['accuracy'])
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir=logs_path)
    save_callback = get_callback(mode)
    model.fit(train_ds, validation_data=val_ds,
              epochs=epochs, verbose=2, callbacks=[save_callback, tensorboard])
    model.save(model_path)
    result = model.evaluate(test_ds)
    print(result)


not_alexnet_model = get_not_alexnet_model()
alexnet_model = get_alexnet_model()
overfitting_model = get_overfitting_model()

simulation('alexnet_model', 100, alexnet_model)
simulation('not_alexnet_model', 100, not_alexnet_model)
simulation('overfitting_model', 100, overfitting_model)