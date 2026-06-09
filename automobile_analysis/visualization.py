import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def chart_avg_price_by_make(avg_price_make: pd.DataFrame) -> go.Figure:
    """Bar chart of average price by manufacturer."""
    return px.bar(
        avg_price_make,
        x="make",
        y="price",
        title="Average Price by Make (Top 10)",
        labels={"price": "Avg Price ($)", "make": "Manufacturer"},
        color_discrete_sequence=["#4A90E2"],
    )


def chart_body_style_distribution(body_style_dist: pd.DataFrame) -> go.Figure:
    """Donut chart of body style distribution."""
    return px.pie(
        body_style_dist,
        values="count",
        names="body-style",
        title="Body Style Distribution",
        hole=0.4,
    )


def chart_engine_size_vs_price(df: pd.DataFrame) -> go.Figure:
    """Scatter plot with OLS trendline for engine size vs price."""
    return px.scatter(
        df,
        x="engine-size",
        y="price",
        trendline="ols",
        title="Engine Size vs Price Correlation",
        labels={"engine-size": "Engine Size (cc)", "price": "Price ($)"},
    )


def chart_fuel_type_distribution(fuel_dist: pd.DataFrame) -> go.Figure:
    """Pie chart of fuel type distribution."""
    return px.pie(
        fuel_dist,
        values="count",
        names="fuel-type",
        title="Fuel Type Distribution",
    )


def chart_avg_mpg_by_body_style(mpg_body: pd.DataFrame) -> go.Figure:
    """Grouped bar chart of city/highway MPG by body style."""
    return px.bar(
        mpg_body,
        x="body-style",
        y=["city-mpg", "highway-mpg"],
        barmode="group",
        title="Average MPG by Body Style",
    )
