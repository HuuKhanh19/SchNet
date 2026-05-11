"""
Phase 1.1b - Diagnose Checkpoint File

Identifies what format the checkpoint actually is by examining file head bytes.
Tries multiple load methods to determine if file is corrupted, wrong format,
or just needs different loader.

Run:  python 01b_diagnose_checkpoint.py <checkpoint_path>
"""

import sys
from pathlib import Path

import torch


def header(s):
    print(f"\n{'=' * 64}\n {s}\n{'=' * 64}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python 01b_diagnose_checkpoint.py <checkpoint_path>")
        return 1

    path = Path(sys.argv[1])
    if not path.exists():
        print(f"[FAIL] File not found: {path}")
        return 1

    size_bytes = path.stat().st_size
    size_mb = size_bytes / 1e6

    header("File Inspection")
    print(f"Path: {path}")
    print(f"Size: {size_mb:.2f} MB ({size_bytes:,} bytes)")

    # Read file head
    with open(path, "rb") as f:
        head = f.read(128)

    print(f"\nFirst 8 bytes (hex): {head[:8].hex()}")
    print(f"First 16 bytes (hex): {head[:16].hex()}")
    try:
        head_str = head[:64].decode("utf-8", errors="replace")
        print(f"First 64 bytes as text: {head_str!r}")
    except Exception:
        pass

    # Identify format from magic bytes
    header("Format Detection")
    fmt = "unknown"
    fix_advice = []

    if head[:4] == b"PK\x03\x04":
        fmt = "ZIP (modern PyTorch checkpoint)"
        print("[INFO] File starts with ZIP magic bytes (PK\\x03\\x04).")
        print("       This IS the right format for modern torch.load().")
        print("       If load still fails, the file is TRUNCATED or CORRUPTED.")
        fix_advice = [
            "1. Re-download the checkpoint (current file likely incomplete).",
            "2. Verify checksum if source provides one.",
            "3. Check available disk space during download.",
        ]
    elif head[:2] in (b"\x80\x02", b"\x80\x04", b"\x80\x05"):
        fmt = "Legacy Python pickle (old PyTorch < 1.6)"
        print("[INFO] File is a legacy pickle. torch.load() should auto-detect.")
        fix_advice = [
            "Try: torch.load(path, map_location='cpu', pickle_module=pickle)",
        ]
    elif head.startswith(b"version https://git-lfs.github.com"):
        fmt = "Git LFS POINTER (file not actually downloaded!)"
        print("[!!!] This is just a Git LFS pointer text file, NOT actual weights.")
        fix_advice = [
            "1. cd into the repo directory",
            "2. Run: git lfs install",
            "3. Run: git lfs pull",
            "OR download checkpoint directly via wget/browser from source.",
        ]
    elif head[:1] in (b"{", b"["):
        fmt = "JSON (likely error page or wrong file)"
        fix_advice = [
            "File looks like JSON, not a checkpoint.",
            "Check the original download URL - might be HTML/JSON error response.",
        ]
    elif head[:8] == b"safetensors" or b"safetensors" in head[:64]:
        fmt = "SafeTensors format"
        fix_advice = [
            "Install safetensors: pip install safetensors",
            "Load: from safetensors.torch import load_file; state = load_file(path)",
        ]
    elif head.startswith(b"<!DOCTYPE") or head.startswith(b"<html"):
        fmt = "HTML page (download failed, got error page)"
        fix_advice = [
            "The download returned an HTML error page, not the file.",
            "Re-download from a working URL.",
        ]

    print(f"\nDetected format: {fmt}")

    # Try various load methods
    header("Load Attempts")

    print("\n[Try 1] torch.load(weights_only=False)")
    try:
        state = torch.load(str(path), map_location="cpu", weights_only=False)
        print("  [OK] Loaded successfully!")
        analyze_state_dict(state)
        return 0  # Success
    except Exception as e:
        print(f"  [FAIL] {type(e).__name__}: {str(e)[:200]}")

    print("\n[Try 2] torch.load(weights_only=True)")
    try:
        state = torch.load(str(path), map_location="cpu", weights_only=True)
        print("  [OK] Loaded with weights_only=True")
        analyze_state_dict(state)
        return 0
    except Exception as e:
        print(f"  [FAIL] {type(e).__name__}: {str(e)[:200]}")

    print("\n[Try 3] safetensors.torch.load_file")
    try:
        from safetensors.torch import load_file
        state = load_file(str(path))
        print("  [OK] Loaded as safetensors!")
        analyze_state_dict(state)
        return 0
    except ImportError:
        print("  [SKIP] safetensors not installed (pip install safetensors)")
    except Exception as e:
        print(f"  [FAIL] {type(e).__name__}: {str(e)[:200]}")

    # All failed
    header("Diagnosis & Fix")
    print(f"Format: {fmt}")
    print(f"Size: {size_mb:.1f} MB")
    print()
    if fix_advice:
        print("Recommended actions:")
        for a in fix_advice:
            print(f"  {a}")
    else:
        print("File format unrecognized. Suggestions:")
        print("  1. Re-download from original source")
        print("  2. Check if format is documented in source README")
        print("  3. Compare file size with expected (UniMol2-570M ~1.1 GB at bf16)")

    print("\nExpected sizes for UniMol2 variants (encoder + heads):")
    print("  unimol2_84M:    ~340 MB (fp32) /  170 MB (bf16)")
    print("  unimol2_164M:   ~660 MB (fp32) /  330 MB (bf16)")
    print("  unimol2_310M:  ~1240 MB (fp32) /  620 MB (bf16)  <-- closest to your 672 MB!")
    print("  unimol2_570M:  ~2270 MB (fp32) / 1140 MB (bf16)")
    print("  unimol2_1100M: ~4400 MB (fp32) / 2200 MB (bf16)")

    return 1


def analyze_state_dict(state):
    """If load succeeded, examine the state dict structure."""
    print("\n  State dict analysis:")
    if isinstance(state, dict):
        print(f"    Top-level type: dict with {len(state)} keys")
        top_keys = list(state.keys())[:10]
        print(f"    First keys: {top_keys}")

        # Look for nested 'model' key (common in UniMol/UniCore checkpoints)
        if "model" in state:
            sd = state["model"]
            print(f"    Found 'model' subkey. State_dict has {len(sd)} weight tensors.")
            if sd:
                first_key = next(iter(sd.keys()))
                first_val = sd[first_key]
                if torch.is_tensor(first_val):
                    print(f"    Sample tensor: {first_key}, shape={tuple(first_val.shape)}, dtype={first_val.dtype}")
            n_params = sum(v.numel() for v in sd.values() if torch.is_tensor(v))
            print(f"    Total params: {n_params/1e6:.1f}M")
        elif "state_dict" in state:
            sd = state["state_dict"]
            print(f"    Found 'state_dict' subkey. {len(sd)} tensors.")
            n_params = sum(v.numel() for v in sd.values() if torch.is_tensor(v))
            print(f"    Total params: {n_params/1e6:.1f}M")
        else:
            # State dict might be flat
            tensor_keys = [k for k, v in state.items() if torch.is_tensor(v)]
            if tensor_keys:
                print(f"    Flat state dict with {len(tensor_keys)} tensors (no 'model' wrapper)")
                n_params = sum(state[k].numel() for k in tensor_keys)
                print(f"    Total params: {n_params/1e6:.1f}M")


if __name__ == "__main__":
    sys.exit(main())