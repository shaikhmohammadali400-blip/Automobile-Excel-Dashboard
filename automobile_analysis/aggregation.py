import pandas as pd


def avg_price_by_make(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Return top N manufacturers by average price, descending."""
    return (
        df.groupby("make")["price"]
        .mean()
        .sort_values(ascending=False)
        .head(top_n)
        .reset_index()
    )


def body_style_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Return the count of vehicles per body style."""
    return df["body-style"].value_counts().reset_index()


def top_manufacturers(df: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    """Return top N manufacturers by vehicle count."""
    return df["make"].value_counts().head(top_n).reset_index()


def fuel_type_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Return the count of vehicles per fuel type."""
    return df["fuel-type"].value_counts().reset_index()


def avg_mpg_by_body_style(df: pd.DataFrame) -> pd.DataFrame:
    """Return average city and highway MPG grouped by body style."""
    return (
        df.groupby("body-style")[["city-mpg", "highway-mpg"]]
        .mean()
        .reset_index()
    )


def price_range_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Categorise vehicles into price buckets and return counts.

    Buckets:
        Budget     : price < 10 000
        Mid-Range  : 10 000 <= price < 20 000
        Premium    : 20 000 <= price < 30 000
        Luxury     : price >= 30 000
    """
    bins = [0, 10_000, 20_000, 30_000, float("inf")]
    labels = ["Budget", "Mid-Range", "Premium", "Luxury"]
    df = df.copy()
    df["price_range"] = pd.cut(df["price"], bins=bins, labels=labels, right=False)
    return df["price_range"].value_counts().reset_index()
