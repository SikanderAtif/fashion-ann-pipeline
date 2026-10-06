# fashion-ann-pipeline (dev)

Fully-connected ANN on Fashion-MNIST, versioned with Git + DVC (Google Drive remote).

## Reproduce
```bash
git clone https://github.com/SikanderAtif/fashion-ann-pipeline.git
cd fashion-ann-pipeline
python3 -m venv venv && source venv/bin/activate
python -m pip install tensorflow dvc "dvc[gdrive]" pyyaml scikit-learn matplotlib pandas
dvc repro                 # rebuild everything locally
# or: dvc pull            # fetch versioned data (needs access to the Google Drive remote)
```
