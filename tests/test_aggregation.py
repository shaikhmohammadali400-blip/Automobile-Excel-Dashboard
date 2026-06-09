import pandas as pd
import pytest

from automobile_analysis.aggregation import (
    avg_mpg_by_body_style,
    avg_price_by_make,
    body_style_distribution,
    fuel_type_distribution,
    price_range_distribution,
    top_manufacturers,
)


class TestAvgPriceByMake:
    def test_returns_dataframe(self, sample_df):
        result = avg_price_by_make(sample_df)
        assert isinstance(result, pd.DataFrame)

    def test_columns(self, sample_df):
        result = avg_price_by_make(sample_df)
        assert list(result.columns) == ["make", "price"]

    def test_descending_order(self, sample_df):
        result = avg_price_by_make(sample_df)
        assert list(result["price"]) == sorted(result["price"], reverse=True)

    def test_top_n_limits_rows(self, sample_df):
        result = avg_price_by_make(sample_df, top_n=2)
        assert len(result) == 2

    def test_correct_values(self, sample_df):
        result = avg_price_by_make(sample_df, top_n=10)
        # bmw average = (16500 + 18920) / 2 = 17710
        bmw_row = result[result["make"] == "bmw"]
        assert bmw_row["price"].iloc[0] == pytest.approx(17710.0)
        # audi average = (17450 + 17710) / 2 = 17580
        audi_row = result[result["make"] == "audi"]
        assert audi_row["price"].iloc[0] == pytest.approx(17580.0)


class TestBodyStyleDistribution:
    def test_returns_all_styles(self, sample_df):
        result = body_style_distribution(sample_df)
        styles = set(result["body-style"])
        assert styles == {"convertible", "sedan", "wagon"}

    def test_counts_sum_to_total(self, sample_df):
        result = body_style_distribution(sample_df)
        assert result["count"].sum() == len(sample_df)

    def test_sedan_is_most_common(self, sample_df):
        result = body_style_distribution(sample_df)
        # value_counts returns descending order
        assert result.iloc[0]["body-style"] == "sedan"


class TestTopManufacturers:
    def test_default_top_5(self, sample_df):
        result = top_manufacturers(sample_df)
        assert len(result) <= 5

    def test_top_1(self, sample_df):
        result = top_manufacturers(sample_df, top_n=1)
        assert len(result) == 1
        # audi and bmw both have 2, so the first should be one of them
        assert result.iloc[0]["make"] in ("audi", "bmw")

    def test_counts_are_correct(self, sample_df):
        result = top_manufacturers(sample_df, top_n=10)
        audi = result[result["make"] == "audi"]
        assert audi["count"].iloc[0] == 2


class TestFuelTypeDistribution:
    def test_fuel_types(self, sample_df):
        result = fuel_type_distribution(sample_df)
        types = set(result["fuel-type"])
        assert types == {"gas", "diesel"}

    def test_counts(self, sample_df):
        result = fuel_type_distribution(sample_df)
        assert result["count"].sum() == 5


class TestAvgMpgByBodyStyle:
    def test_columns(self, sample_df):
        result = avg_mpg_by_body_style(sample_df)
        assert set(result.columns) == {"body-style", "city-mpg", "highway-mpg"}

    def test_sedan_mpg(self, sample_df):
        result = avg_mpg_by_body_style(sample_df)
        sedan = result[result["body-style"] == "sedan"]
        # sedans: city-mpg [24, 18, 23] -> mean 21.67
        assert sedan["city-mpg"].iloc[0] == pytest.approx(
            (24 + 18 + 23) / 3, abs=0.01
        )


class TestPriceRangeDistribution:
    def test_only_mid_range_for_sample(self, sample_df):
        result = price_range_distribution(sample_df)
        non_zero = result[result["count"] > 0]
        labels = set(non_zero["price_range"])
        # All sample prices fall in 10k-20k
        assert "Mid-Range" in labels

    def test_counts_sum_to_total(self, sample_df):
        result = price_range_distribution(sample_df)
        assert result["count"].sum() == len(sample_df)

    def test_budget_bucket(self):
        df = pd.DataFrame({"price": [5000.0, 8000.0, 15000.0, 25000.0, 35000.0]})
        result = price_range_distribution(df)
        budget = result[result["price_range"] == "Budget"]
        assert budget["count"].iloc[0] == 2

    def test_luxury_bucket(self):
        df = pd.DataFrame({"price": [5000.0, 31000.0, 40000.0]})
        result = price_range_distribution(df)
        luxury = result[result["price_range"] == "Luxury"]
        assert luxury["count"].iloc[0] == 2

    def test_does_not_mutate_input(self, sample_df):
        cols_before = list(sample_df.columns)
        price_range_distribution(sample_df)
        assert list(sample_df.columns) == cols_before
