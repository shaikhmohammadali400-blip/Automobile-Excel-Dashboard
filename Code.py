import logging
import sys

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data"
COLUMNS = [
    "symboling", "normalized-losses", "make", "fuel-type", "aspiration", "num-of-doors",
    "body-style", "drive-wheels", "engine-location", "wheel-base", "length", "width",
    "height", "curb-weight", "engine-type", "num-of-cylinders", "engine-size",
    "fuel-system", "bore", "stroke", "compression-ratio", "horsepower", "peak-rpm",
    "city-mpg", "highway-mpg", "price",
]

REQUIRED_COLUMNS = [
    "make", "fuel-type", "body-style", "engine-size",
    "horsepower", "city-mpg", "highway-mpg", "price",
]


def load_data(url: str) -> pd.DataFrame:
    """Fetch the automobile dataset from *url* and return a raw DataFrame.

    Raises
    ------
    RuntimeError
        If the data cannot be fetched or parsed.
    """
    try:
        df = pd.read_csv(url, names=COLUMNS, na_values="?")
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load data from {url}: {exc}"
        ) from exc

    if df.empty:
        raise RuntimeError(f"Dataset loaded from {url} is empty")

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise RuntimeError(
            f"Dataset is missing required columns: {missing}"
        )

    logger.info("Loaded %d rows from %s", len(df), url)
    return df


def _coerce_numeric_column(df: pd.DataFrame, column: str) -> None:
    """Convert *column* to numeric, filling NaNs with the column mean.

    Logs the number of values that could not be parsed and raises
    ``ValueError`` if no valid values remain after coercion.
    """
    original_nans = df[column].isna().sum()
    df[column] = pd.to_numeric(df[column], errors="coerce")
    new_nans = df[column].isna().sum()

    coerced = new_nans - original_nans
    if coerced > 0:
        logger.warning(
            "%d value(s) in '%s' could not be parsed and were set to NaN",
            coerced,
            column,
        )

    col_mean = df[column].mean()
    if pd.isna(col_mean):
        raise ValueError(
            f"Column '{column}' has no valid numeric values after coercion; "
            "cannot compute a fill value"
        )

    filled = df[column].isna().sum()
    if filled > 0:
        logger.info(
            "Filling %d NaN(s) in '%s' with mean %.2f", filled, column, col_mean
        )
    df[column] = df[column].fillna(col_mean)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned copy of *df* with numeric columns coerced and filled."""
    df = df.copy()
    _coerce_numeric_column(df, "price")
    _coerce_numeric_column(df, "horsepower")
    return df


def compute_kpis(df: pd.DataFrame) -> dict:
    """Compute and return dashboard KPIs from a cleaned DataFrame."""
    if df.empty:
        raise ValueError("Cannot compute KPIs on an empty DataFrame")

    kpis = {
        "total_vehicles": len(df),
        "avg_price": df["price"].mean(),
        "avg_hp": df["horsepower"].mean(),
        "avg_city_mpg": df["city-mpg"].mean(),
        "avg_highway_mpg": df["highway-mpg"].mean(),
    }
    logger.info(
        "KPIs — vehicles: %d, avg price: $%,.0f, avg HP: %.0f",
        kpis["total_vehicles"],
        kpis["avg_price"],
        kpis["avg_hp"],
    )
    return kpis


def build_figures(df: pd.DataFrame) -> list[go.Figure]:
    """Create all dashboard figures and return them as a list.

    Raises
    ------
    ValueError
        If the DataFrame is empty or required columns are missing.
    """
    if df.empty:
        raise ValueError("Cannot build figures from an empty DataFrame")

    figures: list[go.Figure] = []

    # Chart A: Average Price by Make (Bar Chart)
    avg_price_make = (
        df.groupby("make")["price"]
        .mean()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )
    fig1 = px.bar(
        avg_price_make,
        x="make",
        y="price",
        title="Average Price by Make (Top 10)",
        labels={"price": "Avg Price ($)", "make": "Manufacturer"},
        color_discrete_sequence=["#4A90E2"],
    )
    figures.append(fig1)

    # Chart B: Body Style Distribution (Pie Chart)
    body_style_dist = df["body-style"].value_counts().reset_index()
    fig2 = px.pie(
        body_style_dist,
        values="count",
        names="body-style",
        title="Body Style Distribution",
        hole=0.4,
    )
    figures.append(fig2)

    # Chart C: Engine Size vs Price (Scatter Plot)
    try:
        fig3 = px.scatter(
            df,
            x="engine-size",
            y="price",
            trendline="ols",
            title="Engine Size vs Price Correlation",
            labels={"engine-size": "Engine Size (cc)", "price": "Price ($)"},
        )
    except ImportError:
        logger.warning(
            "statsmodels is not installed; rendering scatter plot without OLS trendline"
        )
        fig3 = px.scatter(
            df,
            x="engine-size",
            y="price",
            title="Engine Size vs Price Correlation",
            labels={"engine-size": "Engine Size (cc)", "price": "Price ($)"},
        )
    figures.append(fig3)

    # Chart D: Fuel Type Distribution (Pie Chart)
    fuel_dist = df["fuel-type"].value_counts().reset_index()
    fig4 = px.pie(
        fuel_dist,
        values="count",
        names="fuel-type",
        title="Fuel Type Distribution",
    )
    figures.append(fig4)

    # Chart E: Average MPG by Body Style (Grouped Bar)
    mpg_body = (
        df.groupby("body-style")[["city-mpg", "highway-mpg"]].mean().reset_index()
    )
    fig5 = px.bar(
        mpg_body,
        x="body-style",
        y=["city-mpg", "highway-mpg"],
        barmode="group",
        title="Average MPG by Body Style",
    )
    figures.append(fig5)

    return figures


def show_figures(figures: list[go.Figure]) -> None:
    """Render each figure, logging errors instead of crashing."""
    for i, fig in enumerate(figures, start=1):
        try:
            fig.show()
        except Exception as exc:
            logger.error("Could not display figure %d: %s", i, exc)


def main() -> None:
    try:
        df = load_data(DATA_URL)
    except RuntimeError as exc:
        logger.error("%s", exc)
        sys.exit(1)

    try:
        df = clean_data(df)
    except ValueError as exc:
        logger.error("Data cleaning failed: %s", exc)
        sys.exit(1)

    try:
        kpis = compute_kpis(df)
    except ValueError as exc:
        logger.error("KPI computation failed: %s", exc)
        sys.exit(1)

    print(f"Total Vehicles: {kpis['total_vehicles']}")
    print(f"Avg Price: ${kpis['avg_price']:,.0f}")
    print(f"Avg HP: {kpis['avg_hp']:.0f}")

    try:
        figures = build_figures(df)
    except ValueError as exc:
        logger.error("Figure generation failed: %s", exc)
        sys.exit(1)

    show_figures(figures)


if __name__ == "__main__":
    main()
