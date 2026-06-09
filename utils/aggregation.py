from __future__ import annotations

from typing import Sequence

import pandas as pd


def get_distribution(
    df: pd.DataFrame,
    column: str,
    top_n: int | None = None,
) -> pd.DataFrame:
    """Return value-count distribution for *column*.

    Parameters
    ----------
    df : pd.DataFrame
        Source dataframe.
    column : str
        Column whose value counts to compute.
    top_n : int | None
        If given, keep only the *top_n* most frequent values.

    Returns
    -------
    pd.DataFrame
        Two-column frame with *column* and ``count``.
    """
    counts = df[column].value_counts()
    if top_n is not None:
        counts = counts.head(top_n)
    return counts.reset_index()


def get_grouped_mean(
    df: pd.DataFrame,
    group_col: str,
    value_cols: str | Sequence[str],
    sort_by: str | None = None,
    ascending: bool = False,
    top_n: int | None = None,
) -> pd.DataFrame:
    """Group by *group_col*, compute mean of *value_cols*, and optionally sort/limit.

    Parameters
    ----------
    df : pd.DataFrame
        Source dataframe.
    group_col : str
        Column to group on.
    value_cols : str | Sequence[str]
        Column(s) to average.
    sort_by : str | None
        Column name to sort results by. Defaults to ``None`` (no sort).
    ascending : bool
        Sort order (default descending).
    top_n : int | None
        Keep only the first *top_n* rows after sorting.

    Returns
    -------
    pd.DataFrame
        Aggregated dataframe.
    """
    if isinstance(value_cols, str):
        value_cols = [value_cols]

    result = df.groupby(group_col)[value_cols].mean().reset_index()

    if sort_by is not None:
        result = result.sort_values(sort_by, ascending=ascending)

    if top_n is not None:
        result = result.head(top_n)

    return result
