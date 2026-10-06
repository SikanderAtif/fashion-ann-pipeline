"""Stage 1: download Fashion-MNIST and save raw arrays to data/raw/."""
import os
import numpy as np
from tensorflow import keras

OUT = "data/raw"

if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
    np.save(f"{OUT}/x_train.npy", x_train)
    np.save(f"{OUT}/y_train.npy", y_train)
    np.save(f"{OUT}/x_test.npy", x_test)
    np.save(f"{OUT}/y_test.npy", y_test)
    print(f"Saved raw data to {OUT}: train={x_train.shape}, test={x_test.shape}")
