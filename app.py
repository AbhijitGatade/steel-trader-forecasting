import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.linear_model import LinearRegression
from datetime import date, datetime, timedelta
from dateutil.relativedelta import relativedelta
warnings.filterwarnings('ignore')

def generate_year_month_df(start_year, start_month, end_year, end_month):
    # Create a date range
    date_range = pd.date_range(
        start=f"{start_year}-{start_month:02d}", 
        end=f"{end_year}-{end_month:02d}", 
        freq="MS"  # Start of the month frequency
    )
    # Create DataFrame with year and month
    df = pd.DataFrame({
        'year': date_range.year,
        'month': date_range.month
    })
    return df

def generate_srno_for_month_year(srno, last_month, last_year, month, year):
    last_date = date(last_year, last_month, 1)
    next_date = date(year, month, 1)

    diff = abs(relativedelta(next_date, last_date))  # Correct order
    month_difference = diff.years * 12 + diff.months  # Convert to months

    return srno + month_difference  # Adjusting SR No. based on month difference

ds = pd.read_csv("products.csv")
brands = sorted(ds["brandname"].unique())
categories = sorted(ds["productname"].unique())
products = sorted(ds["name"].unique())

st.title("Steel Traders Stock Forecasting")

# Step 1: Select brand
selected_brand = st.selectbox("Select brand:", brands)

# Get the corresponding brand ID
selected_brandid = ds[ds["brandname"] == selected_brand]["brandid"].values[0]

# Step 2: Get product categories for selected brand
productcategories = ds[ds["brandid"] == selected_brandid]["productname"].unique()
selected_productcategory = st.selectbox("Select product category:", productcategories)

# Step 3: Get products for selected brand and category
products = ds[(ds["brandid"] == selected_brandid) & (ds["productname"] == selected_productcategory)][["pwid", "name"]]
selected_product = st.selectbox("Select product:", sorted(products["name"].unique()))

selected_productid = products[ds["name"] == selected_product]["pwid"].values[0]

current_year = 2025
years = list(range(current_year, current_year + 5))  # Next 5 years

# Dropdown
selected_year = st.selectbox("Select year:", years)

# Dictionary mapping month names to numbers
months_dict = {
    "January": 1, "February": 2, "March": 3, "April": 4, "May": 5, "June": 6,
    "July": 7, "August": 8, "September": 9, "October": 10, "November": 11, "December": 12
}

# Dropdown with month names
selected_month_name = st.selectbox("Select Month:", list(months_dict.keys()))

# Get the corresponding month number
selected_month = months_dict[selected_month_name]

if st.button("Forecast"):
    filtered_df = ds[(ds["brandid"] == selected_brandid) & (ds["pwid"] == selected_productid)]
    startyear = filtered_df["year"].min()
    startmonth = filtered_df[filtered_df["year"] == startyear]["month"].min()
    endyear = filtered_df["year"].max()
    endmonth = filtered_df[filtered_df["year"] == endyear]["month"].max()

    year_month_df = generate_year_month_df(startyear, startmonth, endyear, endmonth)
    merged_df = pd.merge(year_month_df, filtered_df, on=['year', 'month'], how='left')
    merged_df['weight'] = merged_df['weight'].fillna(0)

    merged_df['brandid'] = selected_brandid
    merged_df['pwid'] = selected_productid
    merged_df['year_month'] = merged_df['year'].astype(str) + '-' + merged_df['month'].astype(str)
    merged_df.insert(0, 'srno', range(1, len(merged_df) + 1))
    merged_df['year'] = merged_df['year'].astype(int)
    merged_df['month'] = merged_df['month'].astype(int)
    merged_df['srno'] = merged_df['srno'].astype(int)

    X = merged_df[["srno"]]
    y = merged_df[["weight"]]


    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(data=merged_df, x='srno', y='weight', marker='o', ax=ax)
    ax.set_title("Month on Month Product Sale From " + str(startmonth) + "/" + str(startyear) + " To " + str(endmonth) + "/" + str(endyear))
    st.pyplot(fig)

    
    last_year = merged_df.iloc[-1]['year']
    last_month = merged_df.iloc[-1]['month']
    srno = merged_df.iloc[-1]['srno']
    nextsrno = generate_srno_for_month_year(srno, last_month, last_year, selected_month, selected_year)

    model = LinearRegression()
    model.fit(X, y)

    predicted_weight = model.predict([[nextsrno]])

    st.write(f"**Selected Brand:** {selected_brand}")
    st.write(f"**Selected Product Category:** {selected_productcategory}")
    st.write(f"**Selected Product:** {selected_product}")
    st.write(f"**Selected Year:** {selected_year}")
    st.write(f"**Selected Month:** {selected_month}")
    st.write("Predicted Weight:", round(predicted_weight[0,0]), "Kgs")
