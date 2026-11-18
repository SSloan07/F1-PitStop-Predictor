"""Load fitted inputs and reproduce the validation transform from raw v1_3 arrays."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
root = Path(__file__).resolve().parent
repo = root.parents[2]
sys.path.insert(0, str(repo))
from src.f1_pitstop.preprocessing import InputPreprocessor
config = __import__('json').loads((root/'config.json').read_text(encoding='utf-8'))
feature_dir = repo/config['feature_package']
preprocessor = InputPreprocessor.load(root/'preprocessor.json')
validation = np.load(root/'validation_inputs.npz', allow_pickle=False)
metadata = pd.read_csv(root/'validation_metadata.csv.gz')
assert np.array_equal(validation['sample_id'], metadata.sample_id.to_numpy())
sample_ids = validation['sample_id'].astype(np.int64, copy=False)
registry = __import__('json').loads((feature_dir/'feature_registry.json').read_text(encoding='utf-8'))
static_frame = pd.read_csv(feature_dir/'static_features.csv.gz')
static_names = registry['static']
static = static_frame.loc[static_frame.sample_id.isin(sample_ids), static_names].to_numpy(dtype=str)
assert np.array_equal(static_frame.loc[static_frame.sample_id.isin(sample_ids), 'sample_id'].to_numpy(), sample_ids)
mask_names = ['padding_mask','slot_exists_mask','slot_row_missing_mask','temporal_unavailable_mask','feature_missing_mask','mask_reason_codes']
with np.load(feature_dir/'windows.npz', allow_pickle=False) as raw:
    masks = {name: raw[name][sample_ids] for name in mask_names}
    transformed = preprocessor.transform(raw['X_numeric'][sample_ids], raw['X_categorical'][sample_ids], static, masks, preprocessor.to_dict()['schema'])
for name in ['X_numeric','X_dynamic_categorical_ids','X_static_categorical_ids', *mask_names]:
    assert np.array_equal(transformed[name], validation[name]), name
print('validación reproducida:', transformed['X_numeric'].shape, transformed['X_dynamic_categorical_ids'].shape, transformed['X_static_categorical_ids'].shape)
