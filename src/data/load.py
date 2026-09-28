"""Load and combine the UCI Student Performance Math and Portuguese files."""

from __future__ import annotations

import pandas as pd

from src.config import MATH_CSV, POR_CSV


def load_raw_dataset(math_path=MATH_CSV, por_path=POR_CSV) -> pd.DataFrame:
    """Return one table with a `subject` column. Duplicate course rows are kept.

    A student who takes both Math and Portuguese is two academic observations,
    not a data-quality duplicate.
    """
    math_df = pd.read_csv(math_path, sep=";")
    por_df = pd.read_csv(por_path, sep=";")
    math_df["subject"] = "math"
    por_df["subject"] = "portuguese"
    frame = pd.concat([math_df, por_df], ignore_index=True)
    frame.columns = [c.strip() for c in frame.columns]
    return frame
