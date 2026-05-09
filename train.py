
"""Train SchNet (multi-conformer hierarchical) on ESOL/FreeSolv/Lipo/BACE.

All settings come from a YAML config file:

    python train.py --config config.yaml

Pipeline per run:

    1. Load full processed dataset with K conformers per molecule
       (cached at data/processed/<name>/data_K{K}.pt).
    2. Load pre-saved split CSVs at
       data/data_split/<dataset>/seed_<i>/{train,val,test}.csv
    3. Map each split's SMILES -> dataset index, build subsets.
    4. Build SchNet, init lin2.bias to mean(target) for regression.
    5. Train, evaluate on val/test, early-stop on val metric.
    6. (Optional) save best checkpoint when val improves.
"""

from __future__ import annotations

import argparse
import os
import os.path as osp
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, roc_auc_score,
)
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from dataset import DATASET_CONFIG, MoleculeNet3D
from data_utils import GraphDataLoader
from schnet import SchNet


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def get_device(spec: str) -> torch.device:
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)


def build_subsets(
    full_dataset: MoleculeNet3D,
    data_root: str,
    name: str,
    seed: int,
) -> Tuple[List, List, List]:
    """Read pre-saved split CSVs and map SMILES -> dataset index."""
    cfg = DATASET_CONFIG[name]
    smiles_col = cfg["smiles_col"]

    # SMILES -> first index in the processed dataset.
    smi2idx = {}
    for i in range(len(full_dataset)):
        s = full_dataset[i].smiles
        if s not in smi2idx:
            smi2idx[s] = i

    split_dir = osp.join(data_root, "data_split", name, f"seed_{seed}")
    if not osp.isdir(split_dir):
        raise FileNotFoundError(
            f"Split directory not found: {split_dir}\n"
            f"Run `python prepare_splits.py` first."
        )

    subsets = {}
    for sp in ("train", "val", "test"):
        df = pd.read_csv(osp.join(split_dir, f"{sp}.csv"))
        smiles_list = df[smiles_col].astype(str).tolist()
        idx_list = [smi2idx[s] for s in smiles_list if s in smi2idx]
        miss = len(smiles_list) - len(idx_list)
        if miss:
            print(
                f"  {sp}: {miss} SMILES not in processed dataset "
                f"(failed conformer gen during caching)"
            )
        subsets[sp] = [full_dataset[i] for i in idx_list]
    return subsets["train"], subsets["val"], subsets["test"]


# ----------------------------------------------------------------------
# Train / eval steps
# ----------------------------------------------------------------------
def train_one_epoch(model, loader, optimizer, device, task, grad_clip):
    model.train()
    total_loss, total_n = 0.0, 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(
            batch.z, batch.pos, batch.atom_to_conf, batch.num_confs,
        ).view(-1)
        y = batch.y.view(-1).float()
        if task == "regression":
            loss = F.mse_loss(out, y)
        else:  # classification: out is a logit
            loss = F.binary_cross_entropy_with_logits(out, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
        total_n += batch.num_graphs
    return total_loss / total_n


@torch.no_grad()
def evaluate(model, loader, device, task) -> dict:
    model.eval()
    ys, preds = [], []
    for batch in loader:
        batch = batch.to(device)
        out = model(
            batch.z, batch.pos, batch.atom_to_conf, batch.num_confs,
        ).view(-1)
        ys.append(batch.y.view(-1).cpu())
        preds.append(out.cpu())
    y = torch.cat(ys).numpy()
    p = torch.cat(preds).numpy()
    if task == "regression":
        return {
            "rmse": float(np.sqrt(mean_squared_error(y, p))),
            "mae": float(mean_absolute_error(y, p)),
        }
    prob = 1.0 / (1.0 + np.exp(-p))
    return {"auc": float(roc_auc_score(y, prob))}


def save_checkpoint(path, model, optimizer, scheduler, epoch,
                    best_val, best_test, cfg):
    torch.save({
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "epoch": epoch,
        "best_val": best_val,
        "best_test": best_test,
        "config": cfg,
    }, path)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)

    name = cfg["dataset"]
    seed = cfg["split_seed"]
    data_root = cfg["data_root"]
    K = int(cfg.get("K", 5))
    ds_cfg = DATASET_CONFIG[name]
    task = ds_cfg["task"]

    torch.manual_seed(cfg["torch_seed"])
    np.random.seed(cfg["torch_seed"])

    device = get_device(cfg.get("device", "auto"))
    print(f"Device: {device}")
    print(f"Dataset: {name} | Task: {task} | Split seed: {seed} | K={K}")

    # ---- Full processed dataset ----
    full_dataset = MoleculeNet3D.from_dataset_name(
        name, data_root=data_root, K=K
    )
    print(f"Full processed dataset: {len(full_dataset)} molecules")

    # ---- Subsets from pre-saved splits ----
    train_set, val_set, test_set = build_subsets(
        full_dataset, data_root, name, seed
    )
    print(
        f"Subsets — train: {len(train_set)}  "
        f"val: {len(val_set)}  test: {len(test_set)}"
    )

    # ---- DataLoaders ----
    bs = cfg["batch_size"]
    nw = cfg.get("num_workers", 0)
    train_loader = GraphDataLoader(train_set, batch_size=bs, shuffle=True,
                                   num_workers=nw)
    val_loader = GraphDataLoader(val_set, batch_size=bs, num_workers=nw)
    test_loader = GraphDataLoader(test_set, batch_size=bs, num_workers=nw)

    # ---- Model ----
    sch = cfg["schnet"]
    model = SchNet(
        hidden_channels=sch["hidden_channels"],
        num_filters=sch["num_filters"],
        num_interactions=sch["num_interactions"],
        num_gaussians=sch["num_gaussians"],
        cutoff=sch["cutoff"],
        max_num_neighbors=sch["max_num_neighbors"],
        atom_readout=sch.get("atom_readout", "add"),
        conf_readout=sch.get("conf_readout", "mean"),
        interaction_dropout=sch.get("interaction_dropout", 0.0),
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"SchNet parameters: {n_params:,}")
    print(f"interaction_dropout: {sch.get('interaction_dropout', 0.0)}")

    # ---- Output bias init from TRAIN ONLY (regression) ----
    if task == "regression":
        train_ys = torch.cat([d.y.view(-1) for d in train_set])
        mean_target = float(train_ys.mean())
        # n_atoms is the # of REAL atoms (already excludes the K replication).
        mean_n_atoms = float(np.mean([d.n_atoms.item() for d in train_set]))
        print(
            f"Train y: mean={mean_target:.4f}  "
            f"mean atoms/mol={mean_n_atoms:.1f}"
        )
        model.init_output_bias(mean_target, mean_n_atoms)

    # ---- Optimiser & scheduler ----
    optimizer = Adam(model.parameters(), lr=cfg["lr"],
                     weight_decay=cfg["weight_decay"])
    sch_cfg = cfg["scheduler"]
    sched_mode = "min" if task == "regression" else "max"
    scheduler = ReduceLROnPlateau(
        optimizer, mode=sched_mode,
        factor=sch_cfg["factor"], patience=sch_cfg["patience"],
        min_lr=sch_cfg["min_lr"],
    )

    primary = "rmse" if task == "regression" else "auc"
    is_better = (
        (lambda new, best: new < best) if task == "regression"
        else (lambda new, best: new > best)
    )

    # ---- Checkpointing (filename now includes K) ----
    ckpt_path = None
    if cfg.get("save_checkpoints", False):
        ckpt_dir = cfg.get("checkpoint_dir", "./checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_path = osp.join(ckpt_dir, f"{name}_seed{seed}_K{K}_best.pt")
        print(f"Best checkpoint will be saved to: {ckpt_path}")

    # ---- Training loop ----
    best_val, best_test, best_epoch, no_improve = None, None, 0, 0

    for epoch in range(1, cfg["epochs"] + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, device, task, cfg["grad_clip"]
        )
        val_metrics = evaluate(model, val_loader, device, task)
        test_metrics = evaluate(model, test_loader, device, task)
        scheduler.step(val_metrics[primary])

        improved = best_val is None or is_better(val_metrics[primary], best_val)
        if improved:
            best_val = val_metrics[primary]
            best_test = test_metrics
            best_epoch = epoch
            no_improve = 0
            if ckpt_path is not None:
                save_checkpoint(
                    ckpt_path, model, optimizer, scheduler,
                    epoch, best_val, best_test, cfg,
                )
        else:
            no_improve += 1

        lr_now = optimizer.param_groups[0]["lr"]
        flag = "  *" if improved else ""
        print(
            f"Epoch {epoch:03d} | loss {train_loss:.4f} | "
            f"val {val_metrics} | test {test_metrics} | "
            f"lr {lr_now:.2e}{flag}"
        )

        if no_improve >= cfg["patience"]:
            print(
                f"Early stopping at epoch {epoch} "
                f"(no val improvement for {cfg['patience']} epochs)."
            )
            break

    print("\n=== Best result ===")
    print(f"dataset={name}  seed={seed}  K={K}")
    print(f"Best val {primary}: {best_val:.4f}  (epoch {best_epoch})")
    print(f"Test metrics @ best val: {best_test}")
    if ckpt_path is not None:
        print(f"Best checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()