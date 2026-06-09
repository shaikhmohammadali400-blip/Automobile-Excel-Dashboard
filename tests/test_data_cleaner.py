import numpy as np
import pandas as pd

from automobile_analysis.data_cleaner import clean_data, clean_horsepower, clean_price


class TestCleanPrice:
    def test_converts_string_prices_to_float(self, dirty_df):
        result = clean_price(dirty_df)
        assert pd.api.types.is_float_dtype(result["price"])

    def test_fills_nan_with_mean(self, dirty_df):
        result = clean_price(dirty_df)
        assert not result["price"].isna().any()

    def test_non_numeric_strings_become_nan_then_filled(self, dirty_df):
        result = clean_price(dirty_df)
        # "bad" was coerced to NaN then filled with mean
        assert not pd.isna(result["price"].iloc[3])

    def test_does_not_mutate_input(self, dirty_df):
        original_price = dirty_df["price"].copy()
        clean_price(dirty_df)
        pd.testing.assert_series_equal(dirty_df["price"], original_price)

    def test_already_clean_data_unchanged(self, sample_df):
        result = clean_price(sample_df)
        pd.testing.assert_series_equal(
            result["price"], sample_df["price"].astype(float), check_dtype=False
        )

    def test_all_nan_prices_stay_nan(self):
        df = pd.DataFrame({"price": [None, None, None]})
        result = clean_price(df)
        # mean of all-NaN is NaN, so fillna(NaN) keeps NaN
        assert result["price"].isna().all()


class TestCleanHorsepower:
    def test_converts_string_hp_to_float(self, dirty_df):
        result = clean_horsepower(dirty_df)
        assert pd.api.types.is_float_dtype(result["horsepower"])

    def test_fills_nan_with_mean(self, dirty_df):
        result = clean_horsepower(dirty_df)
        assert not result["horsepower"].isna().any()

    def test_does_not_mutate_input(self, dirty_df):
        original_hp = dirty_df["horsepower"].copy()
        clean_horsepower(dirty_df)
        pd.testing.assert_series_equal(dirty_df["horsepower"], original_hp)


class TestCleanData:
    def test_cleans_both_price_and_horsepower(self, dirty_df):
        result = clean_data(dirty_df)
        assert pd.api.types.is_float_dtype(result["price"])
        assert pd.api.types.is_float_dtype(result["horsepower"])
        assert not result["price"].isna().any()
        assert not result["horsepower"].isna().any()

    def test_preserves_other_columns(self, dirty_df):
        result = clean_data(dirty_df)
        pd.testing.assert_series_equal(result["make"], dirty_df["make"])

    def test_clean_data_is_idempotent(self, dirty_df):
        first = clean_data(dirty_df)
        second = clean_data(first)
        pd.testing.assert_frame_equal(first, second)
