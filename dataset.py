

"""Dataset for ESOL / FreeSolv / Lipo / BACE with K conformers per molecule.

Generates K conformers via RDKit `EmbedMultipleConfs(ETKDGv3)` + MMFF94
(UFF fallback) and caches the processed graphs under
`<root>/processed/<name>/data_K{K}.pt`.

If RDKit produces fewer than K conformers (>=1 still produced), the missing
slots are filled by randomly duplicating from the produced set so every
molecule ends up with exactly K conformers. Molecules where RDKit produces
zero conformers are dropped entirely.

Each cached `MultiConfData` object holds:
    z              [K*N]     atomic numbers, repeated K times
    pos            [K*N, 3]  K conformers stacked
    atom_to_conf   [K*N]     LOCAL conformer id (0..K-1) — offset on batching
    num_confs      [1]       always K (after padding)
    n_unique_confs [1]       # of distinct conformers actually produced (≤ K)
                             — informational, useful for diagnostics
    n_atoms        [1]       N (# real atoms, not K*N)
    y              [1]
    smiles         str
"""

from __future__ import annotations

import os
import os.path as osp
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from torch_geometric.data import Data
from tqdm import tqdm

# Silence RDKit's noisy stderr.
RDLogger.DisableLog("rdApp.*")


# ----------------------------------------------------------------------
# Per-dataset config matching your cleaned CSVs.
# ----------------------------------------------------------------------
DATASET_CONFIG = {
    "esol": {
        "csv_file": "esol.csv",
        "smiles_col": "smiles",
        "target_col": "measured",
        "task": "regression",
    },
    "freesolv": {
        "csv_file": "freesolv.csv",
        "smiles_col": "smiles",
        "target_col": "measured",
        "task": "regression",
    },
    "lipo": {
        "csv_file": "lipo.csv",
        "smiles_col": "smiles",
        "target_col": "measured",
        "task": "regression",
    },
    "bace": {
        "csv_file": "bace.csv",
        "smiles_col": "SMILES",
        "target_col": "class",
        "task": "classification",
    },
}


# ----------------------------------------------------------------------
# Multi-conformer Data subclass: tells PyG's Batch how to offset
# `atom_to_conf` when concatenating molecules.
# ----------------------------------------------------------------------
class MultiConfData(Data):
    """Data with K conformers stored flat.

    Override `__inc__` so that when PyG batches B molecules together,
    `atom_to_conf` values get shifted by the cumulative number of conformers
    in earlier graphs — turning a per-molecule LOCAL id (0..K-1) into a
    GLOBAL id (0..total_confs-1).
    """

    def __inc__(self, key, value, *args, **kwargs):
        if key == "atom_to_conf":
            nc = self.num_confs
            if isinstance(nc, torch.Tensor):
                return int(nc.sum().item())
            return int(nc)
        return super().__inc__(key, value, *args, **kwargs)


# ----------------------------------------------------------------------
# K-conformer generation with random-duplicate padding
# ----------------------------------------------------------------------
def smiles_to_kconf_data(
    smiles: str,
    y: float,
    K: int = 5,
    seed: int = 42,
) -> Optional[MultiConfData]:
    """SMILES -> AddHs -> EmbedMultipleConfs(ETKDGv3) -> MMFF94 -> Data.

    Always returns a Data with exactly K conformers (padding via random
    duplication of produced conformers if RDKit returns fewer than K).
    Returns None only if zero conformers are produced.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)

    params = AllChem.ETKDGv3()
    params.randomSeed = seed

    cids = list(AllChem.EmbedMultipleConfs(mol, numConfs=K, params=params))
    if len(cids) == 0:
        # Fallback: random initial coordinates.
        params2 = AllChem.ETKDGv3()
        params2.randomSeed = seed
        params2.useRandomCoords = True
        cids = list(AllChem.EmbedMultipleConfs(mol, numConfs=K, params=params2))
        if len(cids) == 0:
            return None

    # Optimize each conformer.
    for cid in cids:
        try:
            if AllChem.MMFFHasAllMoleculeParams(mol):
                AllChem.MMFFOptimizeMolecule(mol, confId=cid, maxIters=500)
            else:
                AllChem.UFFOptimizeMolecule(mol, confId=cid, maxIters=500)
        except Exception:
            pass

    n_atoms = mol.GetNumAtoms()
    K_produced = len(cids)

    # ----- Random-duplicate padding to always reach K conformers -----
    # `pick[k]` = which produced conformer (0..K_produced-1) to use as slot k.
    if K_produced < K:
        rng = np.random.RandomState(seed)
        extra = rng.choice(K_produced, size=K - K_produced, replace=True)
        pick = list(range(K_produced)) + list(extra)
    else:
        pick = list(range(K))  # in case EmbedMultipleConfs ever returns >K

    # Stack positions: [K * n_atoms, 3]
    pos_flat = torch.zeros(K * n_atoms, 3, dtype=torch.float)
    for k, k_src in enumerate(pick):
        conf = mol.GetConformer(cids[k_src])
        positions = torch.tensor(conf.GetPositions(), dtype=torch.float)
        pos_flat[k * n_atoms:(k + 1) * n_atoms] = positions

    # Atomic numbers: repeat K times.
    z_single = torch.tensor(
        [a.GetAtomicNum() for a in mol.GetAtoms()], dtype=torch.long
    )
    z_flat = z_single.repeat(K)

    # Local conformer index per atom: 0,0,..,0, 1,1,..,1, ..., K-1,...,K-1
    atom_to_conf = torch.arange(
        K, dtype=torch.long
    ).repeat_interleave(n_atoms)

    return MultiConfData(
        z=z_flat,
        pos=pos_flat,
        atom_to_conf=atom_to_conf,
        num_confs=torch.tensor([K], dtype=torch.long),       # always K
        n_unique_confs=torch.tensor([K_produced], dtype=torch.long),
        n_atoms=torch.tensor([n_atoms], dtype=torch.long),
        y=torch.tensor([float(y)], dtype=torch.float),
        smiles=smiles,
    )


# ----------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------
class MoleculeNet3D(Dataset):
    """`torch.utils.data.Dataset` over a list of MultiConfData objects."""

    def __init__(self, data_list: List[MultiConfData]):
        self.data_list = data_list

    def __len__(self) -> int:
        return len(self.data_list)

    def __getitem__(self, idx) -> MultiConfData:
        return self.data_list[idx]

    # ------------------------------------------------------------------
    @classmethod
    def from_csv(
        cls,
        csv_path: str,
        smiles_col: str,
        target_col: str,
        K: int = 5,
        cache_path: Optional[str] = None,
        force_reprocess: bool = False,
    ) -> "MoleculeNet3D":
        """Build (or load) dataset from a CSV with K conformers per molecule."""
        if (
            cache_path is not None
            and osp.exists(cache_path)
            and not force_reprocess
        ):
            print(f"Loading cached dataset from {cache_path}")
            data_list = torch.load(cache_path, weights_only=False)
            return cls(data_list)

        print(f"Processing {csv_path}  (K={K} conformers / molecule)")
        df = pd.read_csv(csv_path)
        if smiles_col not in df.columns or target_col not in df.columns:
            raise KeyError(
                f"CSV {csv_path} is missing '{smiles_col}' or '{target_col}'. "
                f"Found columns: {list(df.columns)}"
            )
        df = df.dropna(subset=[smiles_col, target_col]).reset_index(drop=True)
        smiles_list = df[smiles_col].astype(str).tolist()
        y_list = df[target_col].tolist()

        data_list, n_failed = [], 0
        for smi, y in tqdm(
            list(zip(smiles_list, y_list)),
            desc=f"K={K} conformer gen",
        ):
            d = smiles_to_kconf_data(smi, y, K=K)
            if d is None:
                n_failed += 1
                continue
            data_list.append(d)

        # Stats: how often did we have to pad?
        unique_counts = [int(d.n_unique_confs.item()) for d in data_list]
        print(
            f"Processed {len(data_list)} / {len(smiles_list)} molecules  "
            f"({n_failed} failed: zero conformers produced)"
        )
        if unique_counts:
            n_padded = sum(1 for c in unique_counts if c < K)
            print(
                f"  unique conformers per molecule — "
                f"min={min(unique_counts)}, max={max(unique_counts)}, "
                f"mean={np.mean(unique_counts):.2f}"
            )
            print(
                f"  {n_padded} / {len(data_list)} molecules required "
                f"random duplication to reach K={K}"
            )

        if cache_path is not None:
            os.makedirs(osp.dirname(cache_path), exist_ok=True)
            torch.save(data_list, cache_path)
            print(f"Cached to {cache_path}")

        return cls(data_list)

    @classmethod
    def from_dataset_name(
        cls,
        name: str,
        data_root: str = "./data",
        K: int = 5,
        force_reprocess: bool = False,
    ) -> "MoleculeNet3D":
        if name not in DATASET_CONFIG:
            raise ValueError(
                f"Unknown dataset '{name}'. Choose from {list(DATASET_CONFIG)}."
            )
        cfg = DATASET_CONFIG[name]
        csv_path = osp.join(data_root, cfg["csv_file"])
        cache_path = osp.join(
            data_root, "processed", name, f"data_K{K}.pt"
        )
        return cls.from_csv(
            csv_path=csv_path,
            smiles_col=cfg["smiles_col"],
            target_col=cfg["target_col"],
            K=K,
            cache_path=cache_path,
            force_reprocess=force_reprocess,
        )