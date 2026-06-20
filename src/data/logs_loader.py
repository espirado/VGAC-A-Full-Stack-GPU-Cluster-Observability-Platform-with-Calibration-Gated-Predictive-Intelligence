from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass
class Split:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


def load_logs(csv_path: Path | str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    # Basic cleaning
    if df["timestamp"].dt.tz is None:
        df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")
    df = df.drop_duplicates(subset=["timestamp", "host", "message"]).reset_index(drop=True)
    return df


def stratified_split(df: pd.DataFrame, label_col: str = "category", seed: int = 42) -> Split:
    train_df, tmp = train_test_split(
        df, test_size=0.2, stratify=df[label_col], random_state=seed
    )
    val_df, test_df = train_test_split(
        tmp, test_size=0.5, stratify=tmp[label_col], random_state=seed
    )
    return Split(train=train_df, val=val_df, test=test_df)


