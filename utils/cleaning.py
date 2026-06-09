import pandas as pd


def clean_numeric_column(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Convert a column to numeric and fill missing values with the column mean.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to modify **in place**.
    column : str
        Name of the column to clean.

    Returns
    -------
    pd.DataFrame
        The same dataframe (for chaining convenience).
    """
    df[column] = pd.to_numeric(df[column], errors="coerce")
    df[column] = df[column].fillna(df[column].mean())
    return df
