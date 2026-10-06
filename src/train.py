"""Stage 3: build and train the ANN (Flatten -> Dense ReLU -> Dropout -> Dense 10 softmax)."""
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import yaml
from tensorflow import keras

DATA, OUT = "data/processed", "models"

if __name__ == "__main__":
    p = yaml.safe_load(open("params.yaml"))["train"]
    os.makedirs(OUT, exist_ok=True)
    tf.keras.utils.set_random_seed(p.get("seed", 42))

    x_train, y_train = np.load(f"{DATA}/x_train.npy"), np.load(f"{DATA}/y_train.npy")
    x_val, y_val = np.load(f"{DATA}/x_val.npy"), np.load(f"{DATA}/y_val.npy")

    model = keras.Sequential([
        keras.layers.Input(shape=(28, 28)),
        keras.layers.Flatten(),
        keras.layers.Dense(p["dense_units"], activation="relu"),
        keras.layers.Dropout(p["dropout_rate"]),
        keras.layers.Dense(10, activation="softmax"),
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=p["learning_rate"]),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=p["epochs"],
        batch_size=p["batch_size"],
        verbose=2,
    )
    model.save(f"{OUT}/model.h5")
    pd.DataFrame(history.history).to_csv(f"{OUT}/history.csv", index_label="epoch")
    print("Saved models/model.h5 and models/history.csv")
