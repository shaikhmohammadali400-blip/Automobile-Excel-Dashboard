import pandas as pd


def clean_price(df: pd.DataFrame) -> pd.DataFrame:
    """Convert price to numeric and fill missing values with column mean."""
    df = df.copy()
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["price"] = df["price"].fillna(df["price"].mean())
    return df


def clean_horsepower(df: pd.DataFrame) -> pd.DataFrame:
    """Convert horsepower to numeric and fill missing values with column mean."""
    df = df.copy()
    df["horsepower"] = pd.to_numeric(df["horsepower"], errors="coerce")
    df["horsepower"] = df["horsepower"].fillna(df["horsepower"].mean())
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Run all cleaning steps on the dataframe."""
    df = clean_price(df)
    df = clean_horsepower(df)
    return df
