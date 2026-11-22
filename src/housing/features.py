"""Data contract and feature definitions, shared by training and serving."""

import numpy as np
import pandas as pd

TARGET = "MedHouseVal"

# Feature catalog: name -> (min, max, description). Bounds from analysis/data_profile.ipynb:
# physically possible values, plus the source caps on HouseAge and the target.
CONTRACT: dict[str, tuple[float, float, str]] = {
    "MedInc": (0.0, 20.0, "Median income in block group, $10k"),
    "HouseAge": (0.0, 52.0, "Median house age in block group, years"),
    "AveRooms": (0.0, 200.0, "Average rooms per household"),
    "AveBedrms": (0.0, 50.0, "Average bedrooms per household"),
    "Population": (0.0, 50_000.0, "Block group population"),
    "AveOccup": (0.0, 2_000.0, "Average household members"),
    "Latitude": (32.0, 42.5, "Block group latitude"),
    "Longitude": (-125.0, -114.0, "Block group longitude"),
}
TARGET_BOUNDS = (0.0, 5.00001)
FEATURES = list(CONTRACT)
ENGINEERED = ["BedroomsPerRoom", "RoomsPerPerson"]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derived features. Pure function: same code path at train and serve time."""
    return df.assign(
        BedroomsPerRoom=df["AveBedrms"] / df["AveRooms"].replace(0, np.nan),
        RoomsPerPerson=df["AveRooms"] / df["AveOccup"].replace(0, np.nan),
    ).fillna({"BedroomsPerRoom": 0.0, "RoomsPerPerson": 0.0})


def validate(df: pd.DataFrame, with_target: bool = True) -> list[str]:
    """Return contract violations; empty list means valid."""
    bounds = CONTRACT | ({TARGET: (*TARGET_BOUNDS, "")} if with_target else {})
    missing = sorted(set(bounds) - set(df.columns))
    if missing:
        return [f"missing columns: {missing}"]
    errors = []
    for col, (low, high, _) in bounds.items():
        if df[col].isna().any():
            errors.append(f"{col}: nulls")
        out = int((~df[col].between(low, high)).sum())
        if out:
            errors.append(f"{col}: {out} rows outside [{low}, {high}]")
    return errors
