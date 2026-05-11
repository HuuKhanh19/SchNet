"""
Phase 1.3 - Linear Probe with Ridge Regression

Loads cached embeddings from Phase 1.2, fits ridge regression with closed-form
solution, sweeps λ on validation, picks best, reports test RMSE/MAE/R^2.

This baseline RMSE is the THRESHOLD that EGGROLL must beat in Phase 7+.

Run:  python 03_linear_probe.py --cache-dir ./cache [--split-seed 0]
Output: results_03_linear_probe.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Ridge regression with closed-form solution
# ---------------------------------------------------------------------------
def augment_bias(X):
    """Append a column of ones for bias term. (n, d) -> (n, d+1)"""
    return np.concatenate([X, np.ones((X.shape[0], 1), dtype=X.dtype)], axis=1)


def ridge_fit(X, y, lam):
    """
    Solve: w* = argmin_w ||Xw - y||^2 + lam * ||w||^2
         = (X^T X + lam I)^{-1} X^T y

    Use Cholesky for numerical stability + speed.

    X: (n, d), y: (n,), lam: scalar
    Returns: w (d,)
    """
    n, d = X.shape
    A = X.T @ X + lam * np.eye(d, dtype=X.dtype)
    b = X.T @ y
    # Cholesky decomposition: A = L L^T
    L = np.linalg.cholesky(A)
    # Solve L L^T w = b
    z = np.linalg.solve_triangular(L, b, lower=True) if hasattr(np.linalg, 'solve_triangular') \
        else np.linalg.solve(L, b)  # fallback
    # Better: use scipy.linalg.solve_triangular
    try:
        from scipy.linalg import solve_triangular
        z = solve_triangular(L, b, lower=True)
        w = solve_triangular(L.T, z, lower=False)
    except ImportError:
        # Fallback if scipy unavailable
        w = np.linalg.solve(A, b)
    return w


def loocv_ridge(X, y, lam):
    """
    Closed-form Leave-One-Out Cross-Validation MSE for ridge.

    Computes:
      H = X (X^T X + lam I)^{-1} X^T   (hat matrix)
      h_ii = diagonal of H
      LOO_residual_i = (y_i - y_hat_i) / (1 - h_ii)
      LOO_MSE = mean(LOO_residual^2)

    Returns: (loocv_mse, per_sample_loo_sq_err)
    """
    from scipy.linalg import solve_triangular, cholesky
    n, d = X.shape
    A = X.T @ X + lam * np.eye(d, dtype=X.dtype)
    L = cholesky(A, lower=True)

    # w = A^{-1} X^T y
    z = solve_triangular(L, X.T @ y, lower=True)
    w = solve_triangular(L.T, z, lower=False)

    y_hat = X @ w

    # h_ii = ||L^{-1} x_i^T||^2 = sum over rows of (L^{-1} X^T)
    Z = solve_triangular(L, X.T, lower=False) if False else solve_triangular(L, X.T, lower=True)
    h_diag = (Z ** 2).sum(axis=0)  # (n,)

    # Avoid division by zero (h_ii < 1 always for ridge with lam > 0)
    denom = np.maximum(1.0 - h_diag, 1e-8)
    loo_resid = (y - y_hat) / denom
    per_sample = loo_resid ** 2
    return per_sample.mean(), per_sample, w


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_metrics(y_true, y_pred):
    err = y_pred - y_true
    return {
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mae": float(np.mean(np.abs(err))),
        "r2": float(1 - np.var(err) / (np.var(y_true) + 1e-12)),
        "n": int(len(y_true)),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=str, default="./cache")
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--lambdas", type=float, nargs="+",
                        default=[1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0])
    parser.add_argument("--standardize-y", action="store_true",
                        help="Center and scale y by train statistics (helpful for stable LS).")
    parser.add_argument("--standardize-x", action="store_true",
                        help="Center and scale x columns by train statistics.")
    args = parser.parse_args()

    cache = Path(args.cache_dir)

    # Load embeddings + targets
    print("=" * 64)
    print(" Linear Probe Ridge Regression")
    print("=" * 64)

    splits = {}
    for split in ["train", "val", "test"]:
        emb_path = cache / f"{split}_emb_seed{args.split_seed}.npy"
        y_path = cache / f"{split}_y_seed{args.split_seed}.npy"
        if not emb_path.exists() or not y_path.exists():
            print(f"[FAIL] Missing cache: {emb_path} or {y_path}")
            print("Run 02_extract_embeddings.py first.")
            return 1
        X = np.load(emb_path).astype(np.float64)  # fp64 for LS stability
        y = np.load(y_path).astype(np.float64)
        splits[split] = (X, y)
        print(f"  {split}: X={X.shape}, y={y.shape}, y range [{y.min():.3f}, {y.max():.3f}]")

    X_train, y_train = splits["train"]
    X_val, y_val = splits["val"]
    X_test, y_test = splits["test"]

    # Optional standardization
    if args.standardize_x:
        x_mean = X_train.mean(axis=0)
        x_std = X_train.std(axis=0) + 1e-8
        X_train = (X_train - x_mean) / x_std
        X_val = (X_val - x_mean) / x_std
        X_test = (X_test - x_mean) / x_std
        print(f"  X standardized using train statistics")

    if args.standardize_y:
        y_mean = y_train.mean()
        y_std = y_train.std() + 1e-8
        y_train_s = (y_train - y_mean) / y_std
        y_val_s = (y_val - y_mean) / y_std
        y_test_s = (y_test - y_mean) / y_std
        print(f"  y standardized: mean={y_mean:.3f}, std={y_std:.3f}")
    else:
        y_train_s, y_val_s, y_test_s = y_train, y_val, y_test
        y_mean, y_std = 0.0, 1.0

    # Augment bias
    Xa_train = augment_bias(X_train)
    Xa_val = augment_bias(X_val)
    Xa_test = augment_bias(X_test)

    # =========================================================================
    # Sweep lambda
    # =========================================================================
    print("\n--- Lambda sweep ---")
    print(f"  {'lambda':>10s}  {'train_RMSE':>10s}  {'val_RMSE':>10s}  {'LOO_RMSE':>10s}")
    sweep_results = []
    for lam in args.lambdas:
        # Fit on train
        try:
            loo_mse, _, w = loocv_ridge(Xa_train, y_train_s, lam)
        except Exception as e:
            print(f"  lambda={lam:.4g}: LOO failed ({e})")
            continue

        # Predict on val (in standardized space)
        y_val_pred_s = Xa_val @ w
        # Reverse standardization for reporting
        y_val_pred = y_val_pred_s * y_std + y_mean
        y_train_pred = (Xa_train @ w) * y_std + y_mean

        train_rmse = float(np.sqrt(np.mean((y_train_pred - y_train) ** 2)))
        val_rmse = float(np.sqrt(np.mean((y_val_pred - y_val) ** 2)))
        loo_rmse = float(np.sqrt(loo_mse) * y_std)

        print(f"  {lam:>10.4g}  {train_rmse:>10.4f}  {val_rmse:>10.4f}  {loo_rmse:>10.4f}")
        sweep_results.append({
            "lambda": lam,
            "train_rmse": train_rmse,
            "val_rmse": val_rmse,
            "loo_rmse": loo_rmse,
            "w_norm": float(np.linalg.norm(w)),
        })

    if not sweep_results:
        print("[FAIL] No lambda worked.")
        return 1

    # Pick best by val RMSE
    best = min(sweep_results, key=lambda r: r["val_rmse"])
    print(f"\nBest: lambda={best['lambda']:.4g}, val_RMSE={best['val_rmse']:.4f}")

    # Refit at best lambda and evaluate test
    best_lam = best["lambda"]
    _, _, w_best = loocv_ridge(Xa_train, y_train_s, best_lam)

    y_test_pred = (Xa_test @ w_best) * y_std + y_mean
    y_train_pred = (Xa_train @ w_best) * y_std + y_mean
    y_val_pred = (Xa_val @ w_best) * y_std + y_mean

    train_metrics = compute_metrics(y_train, y_train_pred)
    val_metrics = compute_metrics(y_val, y_val_pred)
    test_metrics = compute_metrics(y_test, y_test_pred)

    # =========================================================================
    # Final report
    # =========================================================================
    print("\n" + "=" * 64)
    print(" FINAL RESULTS — Linear Probe on UniMol2-570M (Lipo)")
    print("=" * 64)
    print(f"Best lambda: {best_lam}")
    print(f"\n{'Split':>8s}  {'RMSE':>8s}  {'MAE':>8s}  {'R^2':>8s}  {'N':>5s}")
    for name, m in [("train", train_metrics), ("val", val_metrics), ("test", test_metrics)]:
        print(f"{name:>8s}  {m['rmse']:>8.4f}  {m['mae']:>8.4f}  {m['r2']:>8.4f}  {m['n']:>5d}")

    print(f"\n>>> BASELINE TO BEAT IN EGGROLL TRAINING: <<<")
    print(f"    Test RMSE = {test_metrics['rmse']:.4f}")
    print(f"    Test R^2  = {test_metrics['r2']:.4f}")
    print(f"    (EGGROLL must reduce test RMSE below {test_metrics['rmse']:.4f})")

    # Save
    out_results = {
        "arch": "unimol2_570M",
        "split_seed": args.split_seed,
        "lambdas_tried": args.lambdas,
        "sweep": sweep_results,
        "best_lambda": best_lam,
        "metrics": {
            "train": train_metrics,
            "val": val_metrics,
            "test": test_metrics,
        },
        "embed_dim": X_train.shape[1],
        "n_train": len(X_train),
        "n_val": len(X_val),
        "n_test": len(X_test),
        "standardize_x": args.standardize_x,
        "standardize_y": args.standardize_y,
        "baseline_test_rmse": test_metrics["rmse"],
    }
    out_path = Path(__file__).parent / f"results_03_linear_probe_seed{args.split_seed}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_results, f, indent=2)
    print(f"\nSaved: {out_path}")

    # Optional: compare with SchNet baseline if available
    schnet_baseline = Path(__file__).parent.parent / "schnet_lipo_baseline.json"
    if schnet_baseline.exists():
        with open(schnet_baseline) as f:
            sb = json.load(f)
        print(f"\nSchNet baseline test RMSE: {sb.get('test_rmse', 'N/A')}")
        print(f"UniMol2-570M linear probe: {test_metrics['rmse']:.4f}")
        if "test_rmse" in sb:
            ratio = test_metrics["rmse"] / sb["test_rmse"]
            print(f"UniMol2 / SchNet ratio: {ratio:.2f}x")

    return 0


if __name__ == "__main__":
    sys.exit(main())