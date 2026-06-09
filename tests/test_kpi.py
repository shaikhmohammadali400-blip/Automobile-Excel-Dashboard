import pandas as pd
import pytest

from automobile_analysis.kpi import (
    avg_city_mpg,
    avg_highway_mpg,
    avg_horsepower,
    avg_price,
    total_vehicles,
)


class TestTotalVehicles:
    def test_returns_row_count(self, sample_df):
        assert total_vehicles(sample_df) == 5

    def test_empty_dataframe(self):
        df = pd.DataFrame(columns=["price"])
        assert total_vehicles(df) == 0


class TestAvgPrice:
    def test_correct_mean(self, sample_df):
        expected = sample_df["price"].mean()
        assert avg_price(sample_df) == pytest.approx(expected)

    def test_single_row(self):
        df = pd.DataFrame({"price": [25000.0]})
        assert avg_price(df) == pytest.approx(25000.0)


class TestAvgHorsepower:
    def test_correct_mean(self, sample_df):
        expected = sample_df["horsepower"].mean()
        assert avg_horsepower(sample_df) == pytest.approx(expected)


class TestAvgCityMpg:
    def test_correct_mean(self, sample_df):
        expected = sample_df["city-mpg"].mean()
        assert avg_city_mpg(sample_df) == pytest.approx(expected)


class TestAvgHighwayMpg:
    def test_correct_mean(self, sample_df):
        expected = sample_df["highway-mpg"].mean()
        assert avg_highway_mpg(sample_df) == pytest.approx(expected)

    def test_single_value(self):
        df = pd.DataFrame({"highway-mpg": [35]})
        assert avg_highway_mpg(df) == pytest.approx(35.0)
