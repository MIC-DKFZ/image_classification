## Saving pre-computed features
Run `extract_cls_and_avg_patch_token.py` to save only the (concatenated) class and average patch tokens.

E.g.:
```bash
python extract_cls_and_avg_patch_token.py env=local data=imagenet model=timm
```

This will save the features at `{config.data_dir}/precomputed_features` in the following format: `agg_{method}_{model}_{dataset}_{split}_size{imgsize}_float{precision}.h5`

The configuration parameters specific to the feature extraction are under `precomputed_features`, defaults:

```yaml
precomputed_features:
  precision: 16
  split: null  # all splits by default
  compression: 4  # HDF5 compression level
```

## Linear Probing from pre-computed features
E.g.:
```bash
python main.py model=linear data=precomputed_features/dinov2_imagenet
```