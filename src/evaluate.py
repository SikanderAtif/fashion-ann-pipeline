"""Stage 4: evaluate on the test set, write metrics.json and a confusion matrix image."""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from tensorflow import keras

CLASSES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
           "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

if __name__ == "__main__":
    os.makedirs("reports", exist_ok=True)
    model = keras.models.load_model("models/model.h5")
    x_test, y_test = np.load("data/processed/x_test.npy"), np.load("data/processed/y_test.npy")

    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    y_pred = np.argmax(model.predict(x_test, verbose=0), axis=1)

    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(8, 8))
    ConfusionMatrixDisplay(cm, display_labels=CLASSES).plot(ax=ax, xticks_rotation=45, colorbar=False)
    fig.tight_layout()
    fig.savefig("reports/confusion_matrix.png", dpi=120)

    json.dump({"test_loss": float(loss), "test_accuracy": float(acc)},
              open("metrics.json", "w"), indent=2)
    print(f"test_loss={loss:.4f} test_accuracy={acc:.4f}")
