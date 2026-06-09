import pandas as pd


def total_vehicles(df: pd.DataFrame) -> int:
    """Return the total number of vehicles in the dataset."""
    return len(df)


def avg_price(df: pd.DataFrame) -> float:
    """Return the mean vehicle price."""
    return float(df["price"].mean())


def avg_horsepower(df: pd.DataFrame) -> float:
    """Return the mean horsepower."""
    return float(df["horsepower"].mean())


def avg_city_mpg(df: pd.DataFrame) -> float:
    """Return the mean city MPG."""
    return float(df["city-mpg"].mean())


def avg_highway_mpg(df: pd.DataFrame) -> float:
    """Return the mean highway MPG."""
    return float(df["highway-mpg"].mean())
