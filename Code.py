import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 1. LOAD DATA
# Note: In the UCI dataset, missing values are often marked as '?'
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data"
columns = [
    "symboling", "normalized-losses", "make", "fuel-type", "aspiration", "num-of-doors",
    "body-style", "drive-wheels", "engine-location", "wheel-base", "length", "width",
    "height", "curb-weight", "engine-type", "num-of-cylinders", "engine-size",
    "fuel-system", "bore", "stroke", "compression-ratio", "horsepower", "peak-rpm",
    "city-mpg", "highway-mpg", "price"
]
df = pd.read_csv(url, names=columns, na_values="?")

# 2. DATA CLEANING
# Fill missing prices with mean and convert to numeric
df['price'] = pd.to_numeric(df['price'], errors='coerce')
df['price'] = df['price'].fillna(df['price'].mean())

df['horsepower'] = pd.to_numeric(df['horsepower'], errors='coerce')
df['horsepower'] = df['horsepower'].fillna(df['horsepower'].mean())

# 3. KPI CALCULATIONS (Matches your Dashboard Cards)
total_vehicles = len(df)
avg_price = df['price'].mean()
avg_hp = df['horsepower'].mean()
avg_city_mpg = df['city-mpg'].mean()
avg_highway_mpg = df['highway-mpg'].mean()

print(f"Total Vehicles: {total_vehicles}")
print(f"Avg Price: ${avg_price:,.0f}")
print(f"Avg HP: {avg_hp:.0f}")

# 4. DATA AGGREGATION (Summary Data Sheet)
# Top 10 Makes by Avg Price
avg_price_make = df.groupby('make')['price'].mean().sort_values(ascending=False).head(10).reset_index()

# Body Style Distribution
body_style_dist = df['body-style'].value_counts().reset_index()

# Top 5 Manufacturers by Count
top_5_makes = df['make'].value_counts().head(5).reset_index()

# Engine Size vs Price Correlation
# (Handled directly in the plot)

# 5. VISUALIZATION (Replicating the Dashboard)

# Chart A: Average Price by Make (Bar Chart)
fig1 = px.bar(avg_price_make, x='make', y='price', 
             title="Average Price by Make (Top 10)",
             labels={'price':'Avg Price ($)', 'make':'Manufacturer'},
             color_discrete_sequence=['#4A90E2'])

# Chart B: Body Style Distribution (Pie Chart)
fig2 = px.pie(body_style_dist, values='count', names='body-style', 
             title="Body Style Distribution",
             hole=0.4)

# Chart C: Engine Size vs Price (Scatter Plot)
fig3 = px.scatter(df, x="engine-size", y="price", 
                 trendline="ols", 
                 title="Engine Size vs Price Correlation",
                 labels={"engine-size": "Engine Size (cc)", "price": "Price ($)"})

# Chart D: Fuel Type Distribution (Pie Chart)
fuel_dist = df['fuel-type'].value_counts().reset_index()
fig4 = px.pie(fuel_dist, values='count', names='fuel-type', 
             title="Fuel Type Distribution")

# Chart E: Average MPG by Body Style (Grouped Bar)
mpg_body = df.groupby('body-style')[['city-mpg', 'highway-mpg']].mean().reset_index()
fig5 = px.bar(mpg_body, x='body-style', y=['city-mpg', 'highway-mpg'], 
             barmode='group', title="Average MPG by Body Style")

# Show all figures (In a real app, you'd use Streamlit to layout these)
fig1.show()
fig2.show()
fig3.show()
fig5.show()
