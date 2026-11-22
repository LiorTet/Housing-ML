"""Bronze -> Silver -> Gold. Each layer is immutable, content-addressed and carries a manifest."""

import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from housing.config import ROOT, Settings
from housing.features import FEATURES, TARGET, validate

N_BINS = 10


@dataclass(frozen=True)
class Manifest:
    layer: str
    version: str
    rows: int
    parent: str
    git_sha: str
    stats: dict[str, list[float]] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path) -> "Manifest":
        return cls(**json.loads(path.read_text()))


def git_sha() -> str:
    out = subprocess.run(["git", "describe", "--always", "--dirty"], cwd=ROOT, capture_output=True)
    return out.stdout.decode().strip() or "nogit"


def _verify(path: Path) -> None:
    data = path / "data.csv"
    if data.exists() and hashlib.sha256(data.read_bytes()).hexdigest()[:12] != path.name:
        raise ValueError(f"{data} does not match its version: modified after commit.")


def _commit(
    settings: Settings, layer: str, df: pd.DataFrame, parent: str, store: bool = True, **stats
) -> Manifest:
    """Version = content hash. Existing versions are never rewritten."""
    content = df.to_csv(index=False).encode()
    manifest = Manifest(
        layer, hashlib.sha256(content).hexdigest()[:12], len(df), parent, git_sha(), stats
    )
    out = settings.data_dir / layer / manifest.version
    if not out.exists():
        out.mkdir(parents=True)
        if store:
            (out / "data.csv").write_bytes(content)
        (out / "manifest.json").write_text(json.dumps(asdict(manifest), indent=2))
    _verify(out)
    return manifest


def bronze(settings: Settings) -> tuple[pd.DataFrame, Manifest]:
    """Source as received: hashed and referenced, never copied or modified."""
    df = pd.read_csv(settings.bronze_path)
    return df, _commit(settings, "bronze", df, settings.bronze_path.name, store=False)


def silver(settings: Settings, df: pd.DataFrame, parent: Manifest) -> tuple[pd.DataFrame, Manifest]:
    """Typed, deduplicated, contract-validated. Violations stop the pipeline."""
    df = df[[*FEATURES, TARGET]].astype("float64").drop_duplicates()
    if errors := validate(df):
        raise ValueError(f"Silver contract violated: {errors}")
    if settings.sample_fraction < 1.0:
        df = df.sample(frac=settings.sample_fraction, random_state=settings.seed)
    df = df.reset_index(drop=True)
    return df, _commit(settings, "silver", df, parent.version)


def gold(settings: Settings, df: pd.DataFrame, parent: Manifest) -> tuple[pd.DataFrame, Manifest]:
    """Model-ready: split by row hash + train deciles (the drift reference).

    A row keeps its split in every Gold version, so a re-scored champion never sees its train rows.
    """
    bucket = pd.util.hash_pandas_object(df, index=False) % 100
    df = df.assign(split=np.where(bucket < settings.test_size * 100, "test", "train"))
    quantiles = np.linspace(0, 1, N_BINS + 1)
    train = df[df["split"] == "train"]
    stats = {c: train[c].quantile(quantiles).round(6).tolist() for c in [*FEATURES, TARGET]}
    return df, _commit(settings, "gold", df, parent.version, **stats)


def build(settings: Settings) -> Manifest:
    """Bronze -> Silver -> Gold, then point CURRENT at the new Gold version."""
    df, manifest = bronze(settings)
    df, manifest = silver(settings, df, manifest)
    _, manifest = gold(settings, df, manifest)
    (settings.data_dir / "gold" / "CURRENT").write_text(manifest.version)
    return manifest


def load_gold(settings: Settings, version: str | None = None) -> tuple[pd.DataFrame, Manifest]:
    path = settings.data_dir / "gold"
    version = version or (path / "CURRENT").read_text().strip()
    _verify(path / version)
    return pd.read_csv(path / version / "data.csv"), Manifest.load(path / version / "manifest.json")


def split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    train, test = df[df["split"] == "train"], df[df["split"] == "test"]
    return train[FEATURES], train[TARGET], test[FEATURES], test[TARGET]
