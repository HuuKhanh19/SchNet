"""Generate 5-seed scaffold splits for ESOL / FreeSolv / Lipo / BACE.

Run once after placing your CSVs in `data/`:

    python prepare_splits.py

For every dataset and every seed in {0,1,2,3,4}, writes:

    data/data_split/{dataset}/seed_{i}/{train,val,test}.csv

Each split file has the same columns as the original CSV. Splits sum to 81/9/10
(`ratio_test=0.10`, `ratio_valid=0.10` of the remaining 90% = 9% of total).

The scaffold split logic mirrors the user's `splitters.random_scaffold_split`
but operates on row indices (so we can save full CSV rows directly).
"""

from __future__ import annotations

import argparse
import os
import os.path as osp
from collections import defaultdict
from typing import List, Tuple

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold

from dataset import DATASET_CONFIG

RDLogger.DisableLog("rdApp.*")


def murcko_scaffold(smiles: str, include_chirality: bool = True) -> str:
    """Bemis–Murcko scaffold SMILES (empty string for non-ring molecules)."""
    return MurckoScaffold.MurckoScaffoldSmiles(
        smiles=smiles, includeChirality=include_chirality
    )


def scaffold_split_indices(
    smiles_list: List[str],
    seed: int,
    ratio_test: float = 0.10,
    ratio_valid: float = 0.10,
) -> Tuple[List[int], List[int], List[int]]:
    """Random scaffold split over a list of SMILES.

    Returns
    -------
    (train_idx, valid_idx, test_idx) : lists of row indices.

    Sizes:
        |test|  = floor(ratio_test * N)
        |valid| = floor(ratio_valid * N * (1 - ratio_test))
        |train| = remainder

    With ratio_test = ratio_valid = 0.10 → 81 / 9 / 10.
    """
    n = len(smiles_list)
    n_test = int(ratio_test * n)
    n_valid = int(ratio_valid * n * (1 - ratio_test))

    # Bucket molecule indices by scaffold.
    scaffolds = defaultdict(list)
    for ind, smi in enumerate(smiles_list):
        scaff = murcko_scaffold(smi, include_chirality=True)
        scaffolds[scaff].append(ind)

    # Shuffle scaffold groups deterministically.
    rng = np.random.RandomState(seed)
    keys = np.array(list(scaffolds.keys()), dtype=object)
    perm = rng.permutation(len(keys))
    scaffold_sets = [scaffolds[keys[i]] for i in perm]

    # Greedy fill: test → valid → train.
    train_idx, valid_idx, test_idx = [], [], []
    for scaff_set in scaffold_sets:
        if len(test_idx) + len(scaff_set) <= n_test:
            test_idx.extend(scaff_set)
        elif len(valid_idx) + len(scaff_set) <= n_valid:
            valid_idx.extend(scaff_set)
        else:
            train_idx.extend(scaff_set)

    assert len(set(train_idx) & set(valid_idx)) == 0
    assert len(set(train_idx) & set(test_idx)) == 0
    assert len(set(valid_idx) & set(test_idx)) == 0
    assert len(train_idx) + len(valid_idx) + len(test_idx) == n
    return train_idx, valid_idx, test_idx


def filter_unparseable(df: pd.DataFrame, smiles_col: str) -> pd.DataFrame:
    """Drop rows whose SMILES RDKit cannot parse (rare but possible)."""
    keep = []
    for s in df[smiles_col].astype(str).tolist():
        keep.append(Chem.MolFromSmiles(s) is not None)
    n_drop = (~np.array(keep)).sum()
    if n_drop:
        print(f"  dropped {n_drop} unparseable SMILES")
    return df.loc[keep].reset_index(drop=True)


def prepare_dataset(name: str, data_root: str, seeds: List[int]) -> None:
    cfg = DATASET_CONFIG[name]
    csv_path = osp.join(data_root, cfg["csv_file"])
    print(f"\n=== {name}  ({csv_path}) ===")

    df = pd.read_csv(csv_path)
    df = df.dropna(subset=[cfg["smiles_col"], cfg["target_col"]]).reset_index(drop=True)
    df = filter_unparseable(df, cfg["smiles_col"])
    print(f"  {len(df)} molecules after cleaning")

    smiles_list = df[cfg["smiles_col"]].astype(str).tolist()

    for seed in seeds:
        tr, va, te = scaffold_split_indices(
            smiles_list, seed=seed, ratio_test=0.10, ratio_valid=0.10
        )
        out_dir = osp.join(data_root, "data_split", name, f"seed_{seed}")
        os.makedirs(out_dir, exist_ok=True)

        df.iloc[tr].to_csv(osp.join(out_dir, "train.csv"), index=False)
        df.iloc[va].to_csv(osp.join(out_dir, "val.csv"), index=False)
        df.iloc[te].to_csv(osp.join(out_dir, "test.csv"), index=False)

        print(
            f"  seed={seed}: train={len(tr):>5}  val={len(va):>4}  "
            f"test={len(te):>4}  →  {out_dir}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4],
        help="Split seeds to generate."
    )
    parser.add_argument(
        "--datasets", type=str, nargs="+",
        default=list(DATASET_CONFIG.keys()),
        help="Datasets to process."
    )
    args = parser.parse_args()

    for name in args.datasets:
        prepare_dataset(name, args.data_root, args.seeds)


if __name__ == "__main__":
    main()