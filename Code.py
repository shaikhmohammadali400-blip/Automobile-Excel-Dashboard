import pandas as pd

from utils import (
    clean_numeric_column,
    create_bar_chart,
    create_pie_chart,
    create_scatter_chart,
    get_distribution,
    get_grouped_mean,
)

# 1. LOAD DATA
# Note: In the UCI dataset, missing values are often marked as '?'
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data"
columns = [
    "symboling",
    "normalized-losses",
    "make",
    "fuel-type",
    "aspiration",
    "num-of-doors",
    "body-style",
    "drive-wheels",
    "engine-location",
    "wheel-base",
    "length",
    "width",
    "height",
    "curb-weight",
    "engine-type",
    "num-of-cylinders",
    "engine-size",
    "fuel-system",
    "bore",
    "stroke",
    "compression-ratio",
    "horsepower",
    "peak-rpm",
    "city-mpg",
    "highway-mpg",
    "price",
]
df = pd.read_csv(url, names=columns, na_values="?")

# 2. DATA CLEANING
clean_numeric_column(df, "price")
clean_numeric_column(df, "horsepower")

# 3. KPI CALCULATIONS (Matches your Dashboard Cards)
total_vehicles = len(df)
avg_price = df["price"].mean()
avg_hp = df["horsepower"].mean()
avg_city_mpg = df["city-mpg"].mean()
avg_highway_mpg = df["highway-mpg"].mean()

print(f"Total Vehicles: {total_vehicles}")
print(f"Avg Price: ${avg_price:,.0f}")
print(f"Avg HP: {avg_hp:.0f}")

# 4. DATA AGGREGATION (Summary Data Sheet)
avg_price_make = get_grouped_mean(df, "make", "price", sort_by="price", top_n=10)
body_style_dist = get_distribution(df, "body-style")
top_5_makes = get_distribution(df, "make", top_n=5)
fuel_dist = get_distribution(df, "fuel-type")
mpg_body = get_grouped_mean(df, "body-style", ["city-mpg", "highway-mpg"])

# 5. VISUALIZATION (Replicating the Dashboard)

# Chart A: Average Price by Make (Bar Chart)
fig1 = create_bar_chart(
    avg_price_make,
    x="make",
    y="price",
    title="Average Price by Make (Top 10)",
    labels={"price": "Avg Price ($)", "make": "Manufacturer"},
    color_sequence=["#4A90E2"],
)

# Chart B: Body Style Distribution (Pie Chart)
fig2 = create_pie_chart(
    body_style_dist,
    values="count",
    names="body-style",
    title="Body Style Distribution",
    hole=0.4,
)

# Chart C: Engine Size vs Price (Scatter Plot)
fig3 = create_scatter_chart(
    df,
    x="engine-size",
    y="price",
    title="Engine Size vs Price Correlation",
    labels={"engine-size": "Engine Size (cc)", "price": "Price ($)"},
    trendline="ols",
)

# Chart D: Fuel Type Distribution (Pie Chart)
fig4 = create_pie_chart(
    fuel_dist,
    values="count",
    names="fuel-type",
    title="Fuel Type Distribution",
)

# Chart E: Average MPG by Body Style (Grouped Bar)
fig5 = create_bar_chart(
    mpg_body,
    x="body-style",
    y=["city-mpg", "highway-mpg"],
    title="Average MPG by Body Style",
    barmode="group",
)

# Show all figures (In a real app, you'd use Streamlit to layout these)
fig1.show()
fig2.show()
fig3.show()
fig5.show()
