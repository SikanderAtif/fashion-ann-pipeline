"""Stage 2: normalize to [0, 1] and split a validation set off the training data."""
import os
import numpy as np
import yaml
from sklearn.model_selection import train_test_split

RAW, OUT = "data/raw", "data/processed"

if __name__ == "__main__":
    params = yaml.safe_load(open("params.yaml"))["preprocess"]
    os.makedirs(OUT, exist_ok=True)

    x_train = np.load(f"{RAW}/x_train.npy").astype("float32")
    y_train = np.load(f"{RAW}/y_train.npy")
    x_test = np.load(f"{RAW}/x_test.npy").astype("float32")
    y_test = np.load(f"{RAW}/y_test.npy")

    mean, std = x_train.mean(), x_train.std()
    x_train = (x_train - mean) / std
    x_test = (x_test - mean) / std
    
    x_tr, x_val, y_tr, y_val = train_test_split(
        x_train, y_train,
        test_size=params["test_size"],
        random_state=params["seed"],
        stratify=y_train,
    )

    for name, arr in [("x_train", x_tr), ("y_train", y_tr), ("x_val", x_val),
                      ("y_val", y_val), ("x_test", x_test), ("y_test", y_test)]:
        np.save(f"{OUT}/{name}.npy", arr)
    print(f"Saved processed data: train={x_tr.shape}, val={x_val.shape}, test={x_test.shape}")
# WIP
