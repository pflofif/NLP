from __future__ import annotations

import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit

Strategy = Literal["group", "stratified", "random"]


def make_splits(
    df: pd.DataFrame,
    strategy: Strategy = "group",
    seed: int = 42,
    val_size: float = 0.10,
    test_size: float = 0.10,
    group_col: str = "place_name",
    target_col: str = "rating",
) -> dict[str, pd.DataFrame]:
    assert val_size + test_size < 1.0, "val_size + test_size must be < 1"

    idx = np.arange(len(df))

    if strategy == "group":
        groups = df[group_col].values

        gss_test = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        trainval_idx, test_idx = next(gss_test.split(idx, groups=groups))

        val_frac_of_trainval = val_size / (1.0 - test_size)
        gss_val = GroupShuffleSplit(n_splits=1, test_size=val_frac_of_trainval, random_state=seed)
        train_idx, val_idx = next(
            gss_val.split(trainval_idx, groups=groups[trainval_idx])
        )
        train_idx = trainval_idx[train_idx]
        val_idx = trainval_idx[val_idx]

    elif strategy == "stratified":
        labels = df[target_col].values

        sss_test = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        trainval_idx, test_idx = next(sss_test.split(idx, labels))

        val_frac_of_trainval = val_size / (1.0 - test_size)
        sss_val = StratifiedShuffleSplit(
            n_splits=1, test_size=val_frac_of_trainval, random_state=seed
        )
        train_idx, val_idx = next(
            sss_val.split(trainval_idx, labels[trainval_idx])
        )
        train_idx = trainval_idx[train_idx]
        val_idx = trainval_idx[val_idx]

    elif strategy == "random":
        rng = np.random.default_rng(seed)
        shuffled = rng.permutation(len(df))
        n_test = int(len(df) * test_size)
        n_val = int(len(df) * val_size)
        test_idx = shuffled[:n_test]
        val_idx = shuffled[n_test: n_test + n_val]
        train_idx = shuffled[n_test + n_val:]

    else:
        raise ValueError(f"Unknown strategy: {strategy!r}. Use 'group', 'stratified', or 'random'.")

    return {
        "train": df.iloc[train_idx].reset_index(drop=True),
        "val": df.iloc[val_idx].reset_index(drop=True),
        "test": df.iloc[test_idx].reset_index(drop=True),
    }


def save_splits(
    splits: dict[str, pd.DataFrame],
    out_dir: Path | str,
    id_col: str = "doc_id",
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, df_split in splits.items():
        ids = df_split[id_col].tolist()
        path = out_dir / f"splits_{name}_ids.txt"
        path.write_text("\n".join(str(i) for i in ids), encoding="utf-8")
        print(f"Saved {len(ids)} ids → {path}")


def make_manifest(
    splits: dict[str, pd.DataFrame],
    strategy: Strategy,
    seed: int,
    group_col: str | None,
    target_col: str | None,
) -> dict:
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "strategy": strategy,
        "group_col": group_col,
        "target_col": target_col,
        "splits": {
            name: {"n_docs": len(df_s), "frac": round(len(df_s) / sum(len(v) for v in splits.values()), 4)}
            for name, df_s in splits.items()
        },
    }


def save_manifest(manifest: dict, path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Manifest saved → {path}")
