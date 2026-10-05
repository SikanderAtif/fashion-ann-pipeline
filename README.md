<<<<<<< HEAD
# fashion-ann-pipeline (typo fixed)
=======
# fashion-ann-pipeline (dev)
>>>>>>> 86596a2 (README: dev title)

Fully-connected ANN on Fashion-MNIST, versioned with Git + DVC (Google Drive remote).

## Reproduce
```
pip install tensorflow dvc "dvc[gdrive]" pyyaml scikit-learn matplotlib pandas
dvc pull
dvc repro
```
