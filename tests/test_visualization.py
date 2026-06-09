import pandas as pd
import plotly.graph_objects as go

from automobile_analysis.aggregation import (
    avg_mpg_by_body_style,
    avg_price_by_make,
    body_style_distribution,
    fuel_type_distribution,
)
from automobile_analysis.visualization import (
    chart_avg_mpg_by_body_style,
    chart_avg_price_by_make,
    chart_body_style_distribution,
    chart_engine_size_vs_price,
    chart_fuel_type_distribution,
)


class TestChartAvgPriceByMake:
    def test_returns_figure(self, sample_df):
        agg = avg_price_by_make(sample_df)
        fig = chart_avg_price_by_make(agg)
        assert isinstance(fig, go.Figure)

    def test_has_title(self, sample_df):
        agg = avg_price_by_make(sample_df)
        fig = chart_avg_price_by_make(agg)
        assert "Average Price" in fig.layout.title.text


class TestChartBodyStyleDistribution:
    def test_returns_figure(self, sample_df):
        agg = body_style_distribution(sample_df)
        fig = chart_body_style_distribution(agg)
        assert isinstance(fig, go.Figure)


class TestChartEngineSizeVsPrice:
    def test_returns_figure(self, sample_df):
        fig = chart_engine_size_vs_price(sample_df)
        assert isinstance(fig, go.Figure)


class TestChartFuelTypeDistribution:
    def test_returns_figure(self, sample_df):
        agg = fuel_type_distribution(sample_df)
        fig = chart_fuel_type_distribution(agg)
        assert isinstance(fig, go.Figure)


class TestChartAvgMpgByBodyStyle:
    def test_returns_figure(self, sample_df):
        agg = avg_mpg_by_body_style(sample_df)
        fig = chart_avg_mpg_by_body_style(agg)
        assert isinstance(fig, go.Figure)
