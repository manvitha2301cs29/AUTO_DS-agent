from __future__ import annotations

import numpy as np
import pandas as pd


def clean_numeric_frame_for_corr(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare a numeric frame for stable correlation calculations."""
    if df.empty:
        return df.copy()

    numeric = df.select_dtypes(include="number").copy()
    if numeric.empty:
        return numeric

    numeric = numeric.replace([np.inf, -np.inf], np.nan)
    usable_cols = [
        col for col in numeric.columns
        if numeric[col].notna().sum() >= 2 and numeric[col].nunique(dropna=True) > 1
    ]
    return numeric[usable_cols]


def safe_corr(a: pd.Series, b: pd.Series) -> float | None:
    """Return a stable Pearson correlation or None if the data is unusable."""
    pair = pd.concat([a, b], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    if len(pair) < 2:
        return None
    if pair.iloc[:, 0].nunique(dropna=True) <= 1 or pair.iloc[:, 1].nunique(dropna=True) <= 1:
        return None
    corr = pair.iloc[:, 0].corr(pair.iloc[:, 1])
    if corr is None or pd.isna(corr):
        return None
    return float(corr)
