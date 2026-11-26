import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# =====================================
# PAGE CONFIGURATION
# =====================================
st.set_page_config(
    page_title="Sales Demand Forecasting System",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Sales Demand Forecasting System")
st.write(
    "Analyze historical sales, compare machine learning models "
    "and forecast future product demand."
)

st.divider()


# =====================================
# HELPER FUNCTIONS
# =====================================
def prepare_monthly_data(df, date_col, sales_col):
    data = df[[date_col, sales_col]].copy()

    data[date_col] = pd.to_datetime(
        data[date_col], errors="coerce"
    )
    data[sales_col] = pd.to_numeric(
        data[sales_col], errors="coerce"
    )

    data[sales_col] = data[sales_col].replace(
        [np.inf, -np.inf], np.nan
    )

    data = data.dropna()

    if data.empty:
        return pd.DataFrame(columns=["Date", "Sales"])

    data["Month"] = (
        data[date_col].dt.to_period("M").dt.to_timestamp()
    )

    monthly = (
        data.groupby("Month")[sales_col]
        .sum()
        .sort_index()
    )

    all_months = pd.date_range(
        start=monthly.index.min(),
        end=monthly.index.max(),
        freq="MS"
    )

    monthly = monthly.reindex(all_months, fill_value=0)

    return pd.DataFrame({
        "Date": all_months,
        "Sales": monthly.values
    })


def create_features(dates, start=1):
    dates = pd.to_datetime(dates)

    return pd.DataFrame({
        "Time": np.arange(start, start + len(dates)),
        "Month": dates.dt.month,
        "Year": dates.dt.year,
        "Month_Sin": np.sin(2 * np.pi * dates.dt.month / 12),
        "Month_Cos": np.cos(2 * np.pi * dates.dt.month / 12)
    })


def get_models():
    return {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=200,
            random_state=42
        )
    }


def format_number(value):
    return f"{value:,.2f}"
    # =====================================
# FILE UPLOAD
# =====================================
st.sidebar.header("Upload Sales Dataset")

uploaded_file = st.sidebar.file_uploader(
    "Upload CSV File",
    type=["csv"]
)

if uploaded_file is None:
    st.info("Please upload your sales CSV file to continue.")
    st.stop()

try:
    df = pd.read_csv(uploaded_file)
except Exception as e:
    st.error(f"Unable to read CSV file: {e}")
    st.stop()

if df.empty:
    st.error("The uploaded CSV file is empty.")
    st.stop()

st.subheader("Dataset Preview")
st.dataframe(df.head(10), use_container_width=True)

st.write("**Dataset Shape:**", df.shape)
st.write("**Available Columns:**", ", ".join(df.columns))


# =====================================
# COLUMN SELECTION
# =====================================
st.sidebar.header("Column Selection")

date_options = list(df.columns)
numeric_options = [
    col for col in df.columns
    if pd.api.types.is_numeric_dtype(df[col])
]

if not numeric_options:
    st.error("No numeric sales column found in the CSV.")
    st.stop()

date_default = next(
    (i for i, col in enumerate(date_options)
     if "date" in col.lower()),
    0
)

date_col = st.sidebar.selectbox(
    "Select Date Column",
    date_options,
    index=date_default
)

sales_col = st.sidebar.selectbox(
    "Select Sales Column",
    numeric_options
)

forecast_period = st.sidebar.selectbox(
    "Forecast Period (Months)",
    [3, 6, 9, 12],
    index=1
)


# =====================================
# DATA PREPARATION
# =====================================
monthly_data = prepare_monthly_data(
    df, date_col, sales_col
)

if monthly_data.empty:
    st.error(
        "No valid date and sales records found. "
        "Please check your selected columns."
    )
    st.stop()

if len(monthly_data) < 6:
    st.warning(
        "At least 6 months of valid sales data are "
        "recommended for forecasting."
    )
    st.stop()


# =====================================
# BUSINESS ANALYTICS
# =====================================
st.header("📊 Business Analytics Dashboard")

total_sales = monthly_data["Sales"].sum()
average_sales = monthly_data["Sales"].mean()
highest_sales = monthly_data["Sales"].max()
lowest_sales = monthly_data["Sales"].min()

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Sales", format_number(total_sales))
col2.metric("Average Monthly Sales", format_number(average_sales))
col3.metric("Highest Monthly Sales", format_number(highest_sales))
col4.metric("Lowest Monthly Sales", format_number(lowest_sales))

st.subheader("Monthly Sales Trend")

st.line_chart(
    monthly_data.set_index("Date")["Sales"]
)

st.subheader("Monthly Sales Data")
st.dataframe(monthly_data, use_container_width=True)
# =====================================
# MODEL TRAINING AND COMPARISON
# =====================================
st.header("🤖 Machine Learning Model Comparison")

X = create_features(monthly_data["Date"])
y = monthly_data["Sales"].values

test_size = min(6, max(2, len(monthly_data) // 5))

train_X = X.iloc[:-test_size]
test_X = X.iloc[-test_size:]

train_y = y[:-test_size]
test_y = y[-test_size:]

model_results = {}
model_predictions = {}
trained_models = {}

for name, model in get_models().items():
    model.fit(train_X, train_y)

    predictions = np.maximum(model.predict(test_X), 0)

    mae = mean_absolute_error(test_y, predictions)
    rmse = np.sqrt(mean_squared_error(test_y, predictions))

    r2 = (
        r2_score(test_y, predictions)
        if len(test_y) >= 3 else np.nan
    )

    model_results[name] = {
        "MAE": mae,
        "RMSE": rmse,
        "R2 Score": r2
    }

    model_predictions[name] = predictions
    trained_models[name] = model

results_df = pd.DataFrame(model_results).T

st.subheader("Model Performance")
st.dataframe(results_df, use_container_width=True)

best_model_name = min(
    model_results,
    key=lambda name: model_results[name]["MAE"]
)

st.success(f"Best Performing Model: {best_model_name}")

# =====================================
# ACTUAL VS PREDICTED
# =====================================
st.subheader("Actual vs Predicted Sales")

comparison_df = pd.DataFrame({
    "Actual Sales": test_y,
    "Predicted Sales": model_predictions[best_model_name]
})

st.line_chart(comparison_df)

# =====================================
# FUTURE SALES FORECASTING
# =====================================
st.header("🔮 Future Sales Forecast")

best_model = get_models()[best_model_name]

best_model.fit(X, y)

last_date = monthly_data["Date"].max()

future_dates = pd.date_range(
    start=last_date + pd.offsets.MonthBegin(1),
    periods=forecast_period,
    freq="MS"
)

future_X = create_features(
    pd.Series(future_dates),
    start=len(monthly_data) + 1
)

future_predictions = np.maximum(
    best_model.predict(future_X),
    0
)

forecast_df = pd.DataFrame({
    "Date": future_dates,
    "Forecasted Sales": future_predictions
})

st.dataframe(forecast_df, use_container_width=True)

st.subheader("Future Demand Forecast Graph")

st.line_chart(
    forecast_df.set_index("Date")["Forecasted Sales"]
)

# =====================================
# DOWNLOAD FORECAST
# =====================================
csv_data = forecast_df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="📥 Download Forecast CSV",
    data=csv_data,
    file_name="sales_forecast.csv",
    mime="text/csv"
)

# =====================================
# BUSINESS RECOMMENDATIONS
# =====================================
st.header("💡 Business Recommendations")

if future_predictions.mean() > average_sales:
    st.write(
        "• Future sales are expected to be higher than "
        "average historical sales."
    )
    st.write(
        "• Consider maintaining sufficient inventory "
        "to meet expected demand."
    )
else:
    st.write(
        "• Future sales are expected to remain below "
        "or near the historical average."
    )
    st.write(
        "• Review inventory levels and promotional strategies."
    )

st.write(
    "• Use historical sales trends to support business decisions."
)

st.caption(
    "Sales Demand Forecasting System | "
    "Developed using Python, Streamlit and Machine Learning"
)
