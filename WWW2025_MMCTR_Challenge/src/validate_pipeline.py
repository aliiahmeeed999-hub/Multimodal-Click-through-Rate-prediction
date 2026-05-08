"""
Pipeline checks for MM-CTR (MicroLens + FuxiCTR).

Usage (from repo root: WWW2025_MMCTR_Challenge/)::

    python src/validate_pipeline.py
"""
from __future__ import annotations

import glob
import importlib
import os
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ERRORS: list[str] = []
WARNINGS: list[str] = []


def check(name: str, condition: bool, fix_hint: str = "") -> None:
    if condition:
        print(f"  ✅ {name}")
    else:
        print(f"  ❌ {name}")
        ERRORS.append(f"{name}. Fix: {fix_hint}")


def warn(name: str, condition: bool, msg: str = "") -> None:
    if not condition:
        print(f"  ⚠️  {name}: {msg}")
        WARNINGS.append(f"{name}: {msg}")
    else:
        print(f"  ✅ {name}")


def _embedding_dim_and_nan(emb: pd.DataFrame) -> tuple[int, float]:
    if "item_id" not in emb.columns:
        return 0, float("nan")
    emb_cols = [c for c in emb.columns if c != "item_id" and pd.api.types.is_numeric_dtype(emb[c])]
    if emb_cols:
        nan_count = float(emb[emb_cols].isna().sum().sum())
        return len(emb_cols), nan_count
    for c in emb.columns:
        if c == "item_id":
            continue
        sample = emb[c].dropna().iloc[0] if len(emb) else None
        if sample is None:
            continue
        if hasattr(sample, "__len__") and not isinstance(sample, (str, bytes)):
            try:
                mat = np.stack(emb[c].values)
                nan_count = float(np.isnan(mat.astype(np.float64)).sum())
                return int(mat.shape[1]), nan_count
            except (ValueError, TypeError):
                continue
    return 0, float("nan")


def _verify_submission_zips() -> None:
    print("\n" + "=" * 50)
    print("STAGE 7: Submission zip (optional)")
    print("=" * 50)
    zips = sorted(glob.glob(str(ROOT / "submission" / "*.zip")))
    if not zips:
        warn("submission/*.zip present", False, "No zip yet — run prediction.py after training.")
        return
    zpath = zips[-1]
    print(f"  Checking: {zpath}")
    try:
        with zipfile.ZipFile(zpath) as zf:
            names = zf.namelist()
            print(f"  Files in zip: {names}")
            csv_names = [n for n in names if n.lower().endswith(".csv")]
            check("zip contains a CSV", len(csv_names) > 0, "Expected prediction.csv inside zip")
            if not csv_names:
                return
            with zf.open(csv_names[0]) as f:
                sub = pd.read_csv(f)
            print(f"  Submission shape: {sub.shape}")
            print(f"  Columns: {sub.columns.tolist()}")
            print(sub.head())
            lower = {c.lower() for c in sub.columns}
            ok_cols = ("id" in lower and "task1" in lower) or len(sub.columns) >= 2
            check("submission columns look valid", ok_cols, "Expected ID + Task1 (or similar)")
            check("no NaN in submission", sub.isnull().sum().sum() == 0, "Fill or drop NaNs")
            print("  ✅ submission CSV sanity")
    except Exception as exc:  # noqa: BLE001
        ERRORS.append(f"Submission zip check failed: {exc}")
        print(f"  ❌ {exc}")


def main() -> None:
    print("\n" + "=" * 50)
    print("STAGE 1: Package Imports")
    print("=" * 50)
    for pkg in ["torch", "fuxictr", "pandas", "numpy", "sklearn", "pyarrow"]:
        try:
            m = importlib.import_module(pkg)
            ver = getattr(m, "__version__", "?")
            print(f"  ✅ {pkg} ({ver})")
        except ImportError as e:
            print(f"  ❌ {pkg}: {e}")
            ERRORS.append(f"Missing package: {pkg}")

    print("\n" + "=" * 50)
    print("STAGE 2: Data Files")
    print("=" * 50)
    required_files = [
        "data/MicroLens_1M_x1/train.parquet",
        "data/MicroLens_1M_x1/valid.parquet",
        "data/MicroLens_1M_x1/test.parquet",
        "data/item_emb.parquet",
    ]
    for f in required_files:
        check(f, (ROOT / f).is_file(), f"Download and place at {f}")

    print("\n" + "=" * 50)
    print("STAGE 3: Data Content Validation")
    print("=" * 50)
    train_path = ROOT / "data/MicroLens_1M_x1/train.parquet"
    if train_path.is_file():
        try:
            train = pd.read_parquet(train_path)
            check("train has 'label' column", "label" in train.columns, "Verify column name with EDA")
            check("train has 'user_id' column", "user_id" in train.columns, "Check actual column names")
            check("train has 'item_id' column", "item_id" in train.columns, "Check actual column names")
            check("train label is binary", train["label"].nunique() <= 2, "Label should be 0/1")
            warn(
                "train CTR is reasonable",
                0.01 < float(train["label"].mean()) < 0.5,
                f"CTR={train['label'].mean():.4f}, expected 0.01–0.50",
            )
            check("No NaN in labels", train["label"].isna().sum() == 0, "Drop or fill NaN labels")
            print(f"  ℹ️  Train shape: {train.shape}")
            print(f"  ℹ️  Train CTR: {train['label'].mean():.4f}")
        except Exception as e:
            ERRORS.append(f"Could not load train.parquet: {e}")
            print(f"  ❌ Could not load train.parquet: {e}")
    else:
        warn("train.parquet readable", False, "Skipped (file missing)")

    emb_path = ROOT / "data/item_emb.parquet"
    if emb_path.is_file():
        try:
            emb = pd.read_parquet(emb_path)
            dim, nan_count = _embedding_dim_and_nan(emb)
            check("item_emb has item_id column", "item_id" in emb.columns, "Check actual ID column name")
            check("item_emb has embedding columns", dim > 0, "No numeric or vector column found")
            warn("embedding dimension is 128", dim == 128, f"Found {dim} dims, expected 128 (ok if vector col)")
            check("No NaN in embeddings", nan_count == 0, f"Found {int(nan_count)} NaN values")
            print(f"  ℹ️  Embedding shape: {emb.shape}")
        except Exception as e:
            ERRORS.append(f"Could not load item_emb.parquet: {e}")
            print(f"  ❌ Could not load item_emb.parquet: {e}")
    else:
        warn("item_emb readable", False, "Skipped (file missing)")

    print("\n" + "=" * 50)
    print("STAGE 4: Custom Modules")
    print("=" * 50)
    optional = [
        "src.device_utils",
        "src.DIN",
        "src.mmctr_dataloader",
    ]
    for module in optional:
        try:
            importlib.import_module(module)
            print(f"  ✅ {module}")
        except Exception as e:
            print(f"  ❌ {module}: {e}")
            ERRORS.append(f"Cannot import {module}: {e}")

    print("\n" + "=" * 50)
    print("STAGE 5: Config Files")
    print("=" * 50)
    configs = glob.glob(str(ROOT / "config" / "*.yaml"))
    check("At least one top-level config/*.yaml", len(configs) >= 1, "Add model/tuner YAML under config/")
    for c in configs:
        p = Path(c)
        check(f"Config readable: {p.name}", p.stat().st_size > 50, f"Config {c} seems empty")

    print("\n" + "=" * 50)
    print("STAGE 6: Output Directories")
    print("=" * 50)
    for d in ["checkpoints", "submission", "logs"]:
        Path(d).mkdir(parents=True, exist_ok=True)
        check(f"Directory exists: {d}", Path(d).is_dir(), f"mkdir {d}")

    _verify_submission_zips()

    print("\n" + "=" * 50)
    print("FINAL REPORT")
    print("=" * 50)
    if ERRORS:
        print(f"\n❌ {len(ERRORS)} ERRORS found:")
        for i, e in enumerate(ERRORS, 1):
            print(f"  {i}. {e}")
        print("\nFix all errors before running training.")
        sys.exit(1)
    if WARNINGS:
        print(f"\n⚠️  {len(WARNINGS)} WARNINGS (non-blocking):")
        for w in WARNINGS:
            print(f"  - {w}")
        print("\nPipeline is ready but review warnings.")
        sys.exit(0)
    print("\n✅ ALL CHECKS PASSED. Pipeline is ready for training!")
    sys.exit(0)


if __name__ == "__main__":
    main()
