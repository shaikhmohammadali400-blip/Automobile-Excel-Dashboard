import pandas as pd
import pytest


@pytest.fixture()
def sample_df() -> pd.DataFrame:
    """Minimal automobile dataframe used across test modules."""
    return pd.DataFrame(
        {
            "symboling": [3, 1, 2, 0, -1],
            "normalized-losses": [None, 164, 164, None, 137],
            "make": ["alfa-romero", "audi", "audi", "bmw", "bmw"],
            "fuel-type": ["gas", "gas", "gas", "gas", "diesel"],
            "aspiration": ["std", "std", "turbo", "std", "std"],
            "num-of-doors": ["two", "four", "four", "two", "four"],
            "body-style": ["convertible", "sedan", "sedan", "sedan", "wagon"],
            "drive-wheels": ["rwd", "fwd", "fwd", "rwd", "rwd"],
            "engine-location": ["front", "front", "front", "front", "front"],
            "wheel-base": [88.6, 99.8, 99.4, 101.2, 103.5],
            "length": [168.8, 176.6, 176.6, 176.8, 189.0],
            "width": [64.1, 66.2, 66.4, 64.8, 66.9],
            "height": [48.8, 54.3, 54.3, 54.3, 55.7],
            "curb-weight": [2548, 2337, 2824, 2507, 3230],
            "engine-type": ["dohc", "ohc", "ohc", "ohc", "ohc"],
            "num-of-cylinders": ["four", "four", "five", "four", "six"],
            "engine-size": [130, 109, 136, 108, 164],
            "fuel-system": ["mpfi", "mpfi", "mpfi", "mpfi", "mpfi"],
            "bore": [3.47, 3.19, 3.19, 3.50, 3.13],
            "stroke": [2.68, 3.40, 3.40, 2.80, 3.40],
            "compression-ratio": [9.0, 10.0, 8.0, 8.8, 8.0],
            "horsepower": [111, 102, 115, 101, 114],
            "peak-rpm": [5000, 5500, 5500, 5800, 5000],
            "city-mpg": [21, 24, 18, 23, 17],
            "highway-mpg": [27, 30, 22, 29, 20],
            "price": [13495, 17450, 17710, 16500, 18920],
        }
    )


@pytest.fixture()
def dirty_df() -> pd.DataFrame:
    """Dataframe with string/NaN values in price and horsepower columns."""
    return pd.DataFrame(
        {
            "price": ["13495", None, "17710", "bad", "18920"],
            "horsepower": ["111", None, "115", "bad", "114"],
            "make": ["a", "b", "c", "d", "e"],
            "fuel-type": ["gas", "gas", "gas", "gas", "diesel"],
            "body-style": ["sedan", "sedan", "sedan", "sedan", "wagon"],
            "city-mpg": [21, 24, 18, 23, 17],
            "highway-mpg": [27, 30, 22, 29, 20],
            "engine-size": [130, 109, 136, 108, 164],
        }
    )
