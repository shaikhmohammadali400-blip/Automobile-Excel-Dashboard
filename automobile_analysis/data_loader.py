import pandas as pd

COLUMN_NAMES = [
    "symboling", "normalized-losses", "make", "fuel-type", "aspiration",
    "num-of-doors", "body-style", "drive-wheels", "engine-location",
    "wheel-base", "length", "width", "height", "curb-weight", "engine-type",
    "num-of-cylinders", "engine-size", "fuel-system", "bore", "stroke",
    "compression-ratio", "horsepower", "peak-rpm", "city-mpg", "highway-mpg",
    "price",
]

DEFAULT_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data"
)


def load_data(source: str = DEFAULT_URL) -> pd.DataFrame:
    """Load automobile data from a CSV source (URL or file path).

    Missing values marked as '?' in the UCI dataset are converted to NaN.
    """
    return pd.read_csv(source, names=COLUMN_NAMES, na_values="?")
