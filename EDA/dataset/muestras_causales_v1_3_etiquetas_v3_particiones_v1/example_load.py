from pathlib import Path
import numpy as np
import pandas as pd
features = Path('EDA/dataset/muestras_causales_v1_3')
labels = Path('EDA/dataset/muestras_causales_v1_3_etiquetas_v3_particiones_v1')
part = pd.read_csv(labels/'sample_partitions.csv.gz')
target = pd.read_csv(labels/'labels_aligned_v3.csv.gz', low_memory=False)
ids = part.loc[part.split.eq('train') & part.supervised_eligible, 'sample_id'].to_numpy()
with np.load(features/'windows.npz', allow_pickle=False) as z:
    X_num = z['X_numeric'][ids]; X_cat = z['X_categorical'][ids]
y = target.set_index('sample_id').loc[ids, 'PitWithin3Laps'].to_numpy()
print(X_num.shape, X_cat.shape, y.shape)
