from pathlib import Path
import numpy as np
import pandas as pd
root = Path(__file__).resolve().parent
x = np.load(root/'train_inputs.npz', allow_pickle=False)
y = pd.read_csv(root/'train_labels.csv.gz')
train_meta = pd.read_csv(root/'train_metadata.csv.gz')
assert np.array_equal(x['sample_id'], train_meta.sample_id.to_numpy())
eligible = train_meta.supervised_eligible.to_numpy(dtype=bool)
assert y.loc[eligible, 'PitWithin3Laps'].notna().all()
print(x['X_numeric'].shape, x['X_dynamic_categorical_ids'].shape, x['X_static_categorical_ids'].shape)
