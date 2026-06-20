from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd


@dataclass
class Google2019S3Config:
    bucket: str
    prefix: str  # e.g., "google2019/a"
    schema_prefix: Optional[str] = None  # e.g., "google2019/schema"

    @property
    def base_uri(self) -> str:
        return f"s3://{self.bucket}/{self.prefix}".rstrip("/")

    def table_glob(self, table_name: str) -> str:
        return f"{self.base_uri}/{table_name}-*.json.gz"

    def schema_uri(self, schema_file: str) -> Optional[str]:
        if not self.schema_prefix:
            return None
        return f"s3://{self.bucket}/{self.schema_prefix}/{schema_file}"


def list_table_files(config: Google2019S3Config, table_name: str, max_files: Optional[int] = None) -> List[str]:
    import s3fs  # lazy import

    fs = s3fs.S3FileSystem(anon=False)
    pattern = config.table_glob(table_name)
    paths = sorted(fs.glob(pattern))
    if max_files is not None and max_files > 0:
        paths = paths[: int(max_files)]
    return [f"s3://{p}" for p in paths]


def _read_jsonl_gz(paths: Iterable[str], sample_rows_per_file: Optional[int] = None) -> pd.DataFrame:
    # JSON Lines with gzip compression; int64 fields may be encoded as strings
    frames: List[pd.DataFrame] = []
    for p in paths:
        try:
            if sample_rows_per_file and sample_rows_per_file > 0:
                # Use iterator to avoid loading full file when sampling
                it = pd.read_json(p, lines=True, compression="gzip", chunksize=sample_rows_per_file)
                frames.append(next(it))
            else:
                frames.append(pd.read_json(p, lines=True, compression="gzip"))
        except StopIteration:
            continue
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    # Best-effort conversions for common int64-like fields represented as strings
    for c in [
        "time",
        "machine_id",
        "collection_id",
        "instance_id",
        "job_id",
    ]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="ignore")
    return df


def load_tables_sample(
    config: Google2019S3Config,
    table_to_max_files: Dict[str, int],
    sample_rows_per_file: int = 10000,
) -> Dict[str, pd.DataFrame]:
    """
    Load a small sample from each requested table into memory.

    table_to_max_files: e.g., {"instance_usage": 10, "machine_events": 5}
    sample_rows_per_file: rows to read per file to control memory
    """
    results: Dict[str, pd.DataFrame] = {}
    for table, max_files in table_to_max_files.items():
        paths = list_table_files(config, table, max_files=max_files)
        df = _read_jsonl_gz(paths, sample_rows_per_file=sample_rows_per_file)
        results[table] = df
    return results


def quick_eda(df: pd.DataFrame) -> Dict[str, object]:
    if df.empty:
        return {"rows": 0, "columns": [], "dtypes": {}, "head": []}
    dtypes = {k: str(v) for k, v in df.dtypes.items()}
    head = df.head(3).to_dict(orient="records")
    return {
        "rows": int(len(df)),
        "columns": list(df.columns),
        "dtypes": dtypes,
        "head": head,
    }






















