# SchNet on ESOL / FreeSolv / Lipo / BACE

End-to-end pipeline. Code structured so that the model, the data loader, and
the dataset class are local files you can edit directly.

## Project layout

```
project/
├── data/
│   ├── esol.csv
│   ├── freesolv.csv
│   ├── lipo.csv
│   └── bace.csv               # CID, SMILES, class
├── schnet.py                  # SchNet model (vendored from torch_geometric)
├── dataset.py                 # MoleculeNet3D + DATASET_CONFIG + conformer gen
├── data_utils.py              # GraphDataLoader (custom collate_fn)
├── prepare_splits.py          # one-shot script: builds 5-seed scaffold splits
├── splitters.py               # your original file (kept for reference)
├── train.py                   # main training entry point
├── config.yaml                # single config file (edit me)
└── README.md
```

## Install

```bash
pip install torch torch-geometric rdkit pandas scikit-learn tqdm pyyaml
# torch-cluster is required by SchNet's radius_graph:
pip install torch-cluster -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html
```

## Workflow

### 1) Place CSVs in `data/`

Schema expected by `DATASET_CONFIG` (edit if different):

| dataset  | smiles_col | target_col | task           |
|----------|------------|------------|----------------|
| esol     | smiles     | measured   | regression     |
| freesolv | smiles     | measured   | regression     |
| lipo     | smiles     | measured   | regression     |
| bace     | SMILES     | class      | classification |

### 2) Generate scaffold splits (one time)

```bash
python prepare_splits.py
```

Writes for every dataset and every seed in {0,1,2,3,4}:

```
data/data_split/<dataset>/seed_<i>/{train,val,test}.csv
```

Each split file has the same columns as the original CSV. Sizes are 81/9/10
(test = 10% of total, val = 10% of the remaining 90% = 9% of total).

To regenerate a subset:

```bash
python prepare_splits.py --datasets bace lipo --seeds 0 1
```

### 3) Edit `config.yaml`

Pick the dataset, the split seed, and any training/SchNet hyperparameter:

```yaml
dataset:    esol
split_seed: 0
batch_size: 64
lr:         5.0e-4
schnet:
  hidden_channels:   128
  num_interactions:  6
  cutoff:            10.0
  ...
```

### 4) Train

```bash
python train.py --config config.yaml
```

First run on each dataset triggers conformer generation (slowest is Lipo:
a few minutes on CPU). Conformers are cached at
`data/processed/<dataset>/data.pt`; subsequent runs load instantly.

To run all 5 seeds for a dataset, just edit `split_seed: 0..4` between runs
or wrap in a shell loop:

```bash
for s in 0 1 2 3 4; do
  python -c "
import yaml; c=yaml.safe_load(open('config.yaml'))
c['split_seed']=$s
yaml.safe_dump(c, open('config.yaml','w'))
"
  python train.py --config config.yaml
done
```

(Or — cleaner — add a `--split_seed` override flag to `train.py` later.)

## Where to edit the core

- **Model layers** (CFConv, InteractionBlock, GaussianSmearing, ShiftedSoftplus):
  `schnet.py`. Forward pass and message-passing formula live there.
- **Batching logic**: `data_utils.py` → `graph_collate`. The current
  implementation calls `torch_geometric.data.Batch.from_data_list`; replace
  that single function call to fully control batching.
- **Conformer generation**: `dataset.py` → `smiles_to_3d_data`. Switch to
  multiple conformers, different force fields, or different atom features
  here.
- **Loss / metrics / training loop**: `train.py`.

## Notes

- **Target normalisation** (regression): mean/std are computed from the
  *current seed's training set only* and passed to `SchNet(mean=..., std=...)`.
  The model's internal post-processing rescales the output back to the
  original target scale. For BACE (classification) we pass `mean=None, std=None`
  so the model output is a raw logit fed to BCEWithLogitsLoss.
- **Metrics**: regression → RMSE (primary), MAE; classification → ROC-AUC.
- **Optimiser**: Adam(lr) + ReduceLROnPlateau(factor=0.7, patience=10).
  Early stop after `patience` epochs with no val improvement.
- **SMILES matching**: `train.py` matches each split CSV's SMILES against the
  processed dataset by string equality. Molecules whose conformer generation
  failed are silently skipped (their count is logged).