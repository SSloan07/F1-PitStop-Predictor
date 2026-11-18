from pathlib import Path
import pandas as pd
import numpy as np
p=Path('muestras_causales_v1')
index=pd.read_csv(p/'sample_index.csv.gz')
static=pd.read_csv(p/'static_features.csv.gz')
labels=pd.read_csv(p/'labels_aligned.csv.gz')
with np.load(p/'windows.npz',allow_pickle=False) as z:
    X_numeric=z['X_numeric']; X_categorical=z['X_categorical']; masks=z['feature_missing_mask']
print(index.shape, static.shape, labels.shape, X_numeric.shape, X_categorical.shape)
