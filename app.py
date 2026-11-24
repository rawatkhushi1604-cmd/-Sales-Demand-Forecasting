
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# PAGE CONFIGURATION

st.set_page_config(
    page_title="Sales Demand Forecasting System",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Sales Demand Forecasting System")
st.write(
    "A flexible Business Analytics and "
    "Product-wise Demand Forecasting platform."
)

# HELPER FUNCTIONS

def create_features(dates, start_time=1):
    dates = pd.DatetimeIndex(pd.to_datetime(dates))

    time_values = np.arange(
        start_time,
        start_time + len(dates)
    )

    features = pd.DataFrame({
        "Time": time_values,
        "Month": dates.month,
        "Year": dates.year,
        "Month_Sin": np.sin(2 * np.pi * dates.month / 12),
        "Month_Cos": np.cos(2 * np.pi * dates.month / 12)
    })

    return features


def calculate_metrics(actual, predicted):
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))

    if len(actual) >= 2:
        r2 = r2_score(actual, predicted)
    else:
        r2 = np.nan

    return mae, rmse, r2


def get_models():
    return {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=200,
            max_depth=5,
            min_samples_leaf=2,
            random_state=42
        )
    }


def make_monthly_data(data, date_column, value_column):
    temp = data[[date_column, value_column]].copy()

    temp[date_column] = pd.to_datetime(
        temp[date_column],
        errors="coerce"
    )

    temp[value_column] = pd.to_numeric(
        temp[value_column],
        errors="coerce"
    )

    temp = temp.dropna()

    if temp.empty:
        return pd.DataFrame()

    temp["Month"] = (
        temp[date_column]
        .dt.to_period("M")
        .dt.to_timestamp()
    )

    monthly = (
        temp.groupby("Month")[value_column]
        .sum()
        .sort_index()
    )

    all_months = pd.date_range(
        start=monthly.index.min(),
        end=monthly.index.max(),
        freq="MS"
    )

    monthly = monthly.reindex(
        all_months,
        fill_value=0
    )

    return pd.DataFrame({
        "Date": all_months,
        "Value": monthly.values
    })


def train_and_evaluate(monthly_data):
    if len(monthly_data) < 6:
        return None

    X = create_features(monthly_data["Date"])
    y = monthly_data["Value"].values

    test_size = min(6, max(2, len(monthly_data) // 5))

    train_X = X.iloc[:-test_size]
    test_X = X.iloc[-test_size:]

    train_y = y[:-test_size]
    test_y = y[-test_size:]

    results = {}
    predictions = {}

    for name, model in get_models().items():
        model.fit(train_X, train_y)
        predicted = model.predict(test_X)

        mae, rmse, r2 = calculate_metrics(
            test_y,
            predicted
        )

        results[name] = {
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2
        }

        predictions[name] = predicted

    best_model_name = min(
        results,
        key=lambda name: results[name]["MAE"]
    )

    return {
        "results": results,
        "predictions": predictions,
        "best_model": best_model_name,
        "actual": test_y,
        "test_dates": monthly_data["Date"].iloc[-test_size:],
        "train_size": len(train_y)
    }


def forecast_future(monthly_data, model_name, periods):
    X = create_features(monthly_data["Date"])
    y = monthly_data["Value"].values

    model = get_models()[model_name]
    model.fit(X, y)

    last_date = monthly_data["Date"].max()

    future_dates = pd.date_range(
        start=last_date + pd.offsets.MonthBegin(1),
        periods=periods,
        freq="MS"
    )

    future_X = create_features(
        future_dates,
        start_time=len(monthly_data) + 1
    )

    predictions = model.predict(future_X)
    predictions = np.maximum(predictions, 0)

    return pd.DataFrame({
        "Date": future_dates,
        "Predicted Value": predictions
    })
    
# CSV UPLOAD

st.header("📂 Upload Your Dataset")

uploaded_file = st.file_uploader(
    "Choose a CSV file",
    type=["csv"],
    key="main_csv_uploader"
)

if uploaded_file is None:
    st.info("Please upload a CSV dataset to begin.")

    st.write("### Supported Industries")
    st.write(
        "Beauty, Books, Clothing, Electronics, "
        "Home & Kitchen, Sports and more."
    )

    st.stop()

try:
    data = pd.read_csv(uploaded_file)
except Exception as error:
    st.error(f"Unable to read CSV file: {error}")
    st.stop()

if data.empty:
    st.error("The uploaded dataset is empty.")
    st.stop()

data.columns = data.columns.astype(str).str.strip()

st.success("Dataset uploaded successfully!")

# DATASET OVERVIEW

st.header("1. Dataset Overview")

col1, col2, col3 = st.columns(3)

col1.metric("Total Records", f"{len(data):,}")
col2.metric("Total Columns", len(data.columns))
col3.metric(
    "Missing Values",
    int(data.isnull().sum().sum())
)

with st.expander("View Dataset Preview"):
    st.dataframe(
        data.head(20),
        use_container_width=True
    )

with st.expander("View Dataset Information"):
    info_df = pd.DataFrame({
        "Column": data.columns,
        "Data Type": data.dtypes.astype(str).values,
        "Missing Values": data.isnull().sum().values
    })

    st.dataframe(
        info_df,
        use_container_width=True
    )

# COLUMN DETECTION

date_candidates = [
    col for col in data.columns
    if any(word in col.lower() for word in [
        "date", "time", "day"
    ])
]

numeric_candidates = data.select_dtypes(
    include=np.number
).columns.tolist()

category_candidates = [
    col for col in data.columns
    if (
        data[col].dtype == "object"
        or str(data[col].dtype) == "category"
    )
]

if not date_candidates:
    st.error(
        "No date column detected. Please upload "
        "a dataset containing a date column."
    )
    st.stop()

if not numeric_candidates:
    st.error(
        "No numeric column detected. Please upload "
        "a dataset containing Sales or Quantity."
    )
    st.stop()

# SIDEBAR COLUMN SELECTION

st.sidebar.header("⚙️ Dataset Settings")

date_col = st.sidebar.selectbox(
    "Select Date Column",
    date_candidates
)

numeric_default = (
    "Sales" if "Sales" in numeric_candidates
    else numeric_candidates[0]
)

numeric_col = st.sidebar.selectbox(
    "Select Sales / Revenue / Units Column",
    numeric_candidates,
    index=numeric_candidates.index(numeric_default)
)

if category_candidates:
    category_col = st.sidebar.selectbox(
        "Select Category Column",
        ["None"] + category_candidates
    )
else:
    category_col = "None"

# PREPARE DATA

working_data = data.copy()

working_data[date_col] = pd.to_datetime(
    working_data[date_col],
    errors="coerce"
)

working_data[numeric_col] = pd.to_numeric(
    working_data[numeric_col],
    errors="coerce"
)

working_data = working_data.dropna(
    subset=[date_col, numeric_col]
)

if working_data.empty:
    st.error(
        "No valid date and numeric values found. "
        "Please check your selected columns."
    )
    st.stop()
# BUSINESS ANALYTICS

st.header("2. Business Analytics")

total_value = working_data[numeric_col].sum()
average_value = working_data[numeric_col].mean()
maximum_value = working_data[numeric_col].max()
minimum_value = working_data[numeric_col].min()

c1, c2, c3, c4 = st.columns(4)

c1.metric("Total Value", f"{total_value:,.2f}")
c2.metric("Average Value", f"{average_value:,.2f}")
c3.metric("Maximum Value", f"{maximum_value:,.2f}")
c4.metric("Minimum Value", f"{minimum_value:,.2f}")

# MONTHLY BUSINESS TREND

st.subheader("Monthly Business Trend")

business_monthly = make_monthly_data(
    working_data,
    date_col,
    numeric_col
)

if not business_monthly.empty:

    chart_data = business_monthly.set_index("Date")

    st.line_chart(
        chart_data["Value"],
        use_container_width=True
    )

    st.dataframe(
        business_monthly,
        use_container_width=True
    )

# CATEGORY ANALYSIS

st.subheader("Category-wise Business Analysis")

if category_col != "None":

    category_analysis = (
        working_data.groupby(category_col)[numeric_col]
        .sum()
        .sort_values(ascending=False)
        .reset_index()
    )

    st.dataframe(
        category_analysis,
        use_container_width=True
    )

    st.bar_chart(
        category_analysis.set_index(category_col),
        use_container_width=True
    )

else:
    st.info(
        "Select a category column in the sidebar "
        "to view category-wise analysis."
)
# OVERALL SALES / DEMAND FORECASTING

st.header("3. Overall Sales / Demand Forecasting")

st.write(
    f"Forecasting selected numeric column: **{numeric_col}**"
)

monthly_data = make_monthly_data(
    working_data,
    date_col,
    numeric_col
)

if len(monthly_data) < 6:

    st.warning(
        "At least 6 months of monthly data are required "
        "for forecasting."
    )

else:

    evaluation = train_and_evaluate(monthly_data)

    if evaluation is not None:

        # MODEL PERFORMANCE

        st.subheader("Model Performance Comparison")

        results_df = pd.DataFrame(
            evaluation["results"]
        ).T

        st.dataframe(
            results_df.round(2),
            use_container_width=True
        )

        best_model_name = evaluation["best_model"]

        st.success(
            f"Best Performing Model: {best_model_name}"
        )

        best_metrics = evaluation["results"][
            best_model_name
        ]

        m1, m2, m3 = st.columns(3)

        m1.metric(
            "MAE",
            f"{best_metrics['MAE']:,.2f}"
        )

        m2.metric(
            "RMSE",
            f"{best_metrics['RMSE']:,.2f}"
        )

        if np.isfinite(best_metrics["R2"]):
            m3.metric(
                "R² Score",
                f"{best_metrics['R2']:.3f}"
            )
        else:
            m3.metric("R² Score", "Not available")

        # ACTUAL VS PREDICTED

        st.subheader("Actual vs Predicted Values")

        comparison_df = pd.DataFrame({
            "Date": evaluation["test_dates"].values,
            "Actual": evaluation["actual"],
            "Predicted": evaluation["predictions"][
                best_model_name
            ]
        }).set_index("Date")

        st.line_chart(
            comparison_df,
            use_container_width=True
        )

        # FUTURE FORECASTING

        st.subheader("Future Forecasting")

        forecast_period = st.slider(
            "Select Forecast Period (Months)",
            min_value=1,
            max_value=24,
            value=6,
            key="overall_forecast_period"
        )

        future_forecast = forecast_future(
            monthly_data,
            best_model_name,
            forecast_period
        )

        st.dataframe(
            future_forecast.round(2),
            use_container_width=True
        )

        st.line_chart(
            future_forecast.set_index("Date"),
            use_container_width=True
        )

        # DOWNLOAD FORECAST

        overall_csv = future_forecast.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            "⬇️ Download Overall Forecast CSV",
            data=overall_csv,
            file_name="overall_sales_forecast.csv",
            mime="text/csv"
        )
        
    # PRODUCT-WISE FORECASTING

    if (
        date_col != "Not selected"
        and product_col != "Not selected"
        and demand_col != "Not selected"
    ):

        forecast_period = st.slider(
            "Select Future Forecast Period (Months)",
            min_value=1,
            max_value=12,
            value=6,
            key="product_forecast_period"
        )

        if st.button(
            "Generate Product-wise Forecast",
            key="generate_product_forecast"
        ):

            product_data = pd.DataFrame({
                "Date": pd.to_datetime(
                    data[date_col],
                    errors="coerce"
                ),
                "Product": data[product_col].astype(str).str.strip(),
                "Value": pd.to_numeric(
                    data[demand_col],
                    errors="coerce"
                )
            }).dropna()

            product_data = product_data[
                (product_data["Product"] != "")
                & (product_data["Product"].str.lower() != "nan")
            ]

            if product_data.empty:

                st.error(
                    "No valid product data found. "
                    "Please check your selected columns."
                )

            else:

                last_month = (
                    product_data["Date"]
                    .max()
                    .to_period("M")
                    .to_timestamp()
                )

                future_dates = pd.date_range(
                    start=last_month + pd.offsets.MonthBegin(1),
                    periods=forecast_period,
                    freq="MS"
                )

                products = sorted(
                    product_data["Product"].unique()
                )

                product_results = []
                future_results = []
                historical_results = []

                progress = st.progress(0)
            
                for index, product in enumerate(products):

                    product_frame = product_data[
                        product_data["Product"] == product
                    ].copy()

                    product_frame = product_frame.sort_values("Date")

                    product_start = (
                        product_frame["Date"]
                        .min()
                        .to_period("M")
                        .to_timestamp()
                    )

                    product_months = pd.date_range(
                        start=product_start,
                        end=last_month,
                        freq="MS"
                    )

                    monthly_series = (
                        product_frame
                        .set_index("Date")["Value"]
                        .resample("MS")
                        .sum()
                        .reindex(product_months, fill_value=0)
                    )

                    monthly_series = monthly_series.astype(float)

                    historical_results.append(
                        pd.DataFrame({
                            "Date": monthly_series.index,
                            "Product": product,
                            "Actual Demand": monthly_series.values
                        })
                    )

                    if len(monthly_series) < 12:
                        continue

                    model_data = pd.DataFrame({
                        "Time": range(len(monthly_series)),
                        "Demand": monthly_series.values
                    })

                    train = model_data.iloc[:-3]
                    test = model_data.iloc[-3:]

                    lr_model = LinearRegression()

                    rf_model = RandomForestRegressor(
                        n_estimators=200,
                        random_state=42,
                        max_depth=5
                    )

                    lr_model.fit(
                        train[["Time"]],
                        train["Demand"]
                    )

                    rf_model.fit(
                        train[["Time"]],
                        train["Demand"]
                    )

                    lr_test = lr_model.predict(
                        test[["Time"]]
                    )

                    rf_test = rf_model.predict(
                        test[["Time"]]
                    )

                    lr_mae = mean_absolute_error(
                        test["Demand"],
                        lr_test
                    )

                    rf_mae = mean_absolute_error(
                        test["Demand"],
                        rf_test
                    )

                    if lr_mae <= rf_mae:
                        selected_model = LinearRegression()
                        selected_model_name = "Linear Regression"
                    else:
                        selected_model = RandomForestRegressor(
                            n_estimators=200,
                            random_state=42,
                            max_depth=5
                        )
                        selected_model_name = "Random Forest"

                    selected_model.fit(
                        model_data[["Time"]],
                        model_data["Demand"]
        )
                    
                    future_time = pd.DataFrame({
                        "Time": range(
                            len(monthly_series),
                            len(monthly_series) + forecast_period
                        )
                    })

                    predictions = selected_model.predict(
                        future_time[["Time"]]
                    )

                    predictions = np.maximum(predictions, 0)

                    product_forecast = pd.DataFrame({
                        "Date": future_dates,
                        "Product": product,
                        "Predicted Demand": predictions,
                        "Selected Model": selected_model_name
                    })

                    future_results.append(product_forecast)

                    product_results.append({
                        "Product": product,
                        "Model": selected_model_name,
                        "Linear Regression MAE": lr_mae,
                        "Random Forest MAE": rf_mae,
                        "Total Predicted Demand": predictions.sum()
                    })

                    progress.progress(
                        (index + 1) / len(products)
                    )

                progress.empty()

                if len(future_results) == 0:

                    st.warning(
                        "Not enough historical data. "
                        "Each product needs at least 12 months of data."
                    )

                else:

                    final_product_forecast = pd.concat(
                        future_results,
                        ignore_index=True
                    )

                    product_model_results = pd.DataFrame(
                        product_results
                    )

                    historical_demand = pd.concat(
                        historical_results,
                        ignore_index=True
                    )
                    
                    st.subheader("📊 Product-wise Future Demand")

                    st.dataframe(
                        final_product_forecast,
                        use_container_width=True
                    )

                    st.subheader("🏆 Monthly Highest Demand Product")

                    monthly_winners = (
                        final_product_forecast
                        .loc[
                            final_product_forecast.groupby("Date")[
                                "Predicted Demand"
                            ].idxmax()
                        ]
                        .sort_values("Date")
                        .reset_index(drop=True)
                    )

                    st.dataframe(
                        monthly_winners[
                            ["Date", "Product", "Predicted Demand"]
                        ],
                        use_container_width=True
                    )

                    st.subheader("🥇 Overall Product Ranking")

                    product_ranking = (
                        final_product_forecast
                        .groupby("Product")["Predicted Demand"]
                        .sum()
                        .sort_values(ascending=False)
                        .reset_index()
                    )

                    product_ranking.columns = [
                        "Product",
                        "Total Predicted Demand"
                    ]

                    st.dataframe(
                        product_ranking,
                        use_container_width=True
                    )

                    best_product = product_ranking.iloc[0]

                    st.success(
                        f"Highest Forecasted Demand: "
                        f"{best_product['Product']} "
                        f"({best_product['Total Predicted Demand']:,.2f} units)"
                    )

                    st.subheader("📈 Historical Monthly Demand")

                    st.dataframe(
                        historical_demand,
                        use_container_width=True
                    )

                    st.subheader("⚖️ Model Performance Comparison")

                    st.dataframe(
                        product_model_results,
                        use_container_width=True
                    )

                    st.download_button(
                        "⬇️ Download Product Forecast CSV",
                        data=final_product_forecast.to_csv(
                            index=False
                        ).encode("utf-8"),
                        file_name="product_demand_forecast.csv",
                        mime="text/csv"
                    )

                    st.download_button(
                        "⬇️ Download Monthly Winners CSV",
                        data=monthly_winners.to_csv(
                            index=False
                        ).encode("utf-8"),
                        file_name="monthly_highest_demand.csv",
                        mime="text/csv"
                    )

                    st.download_button(
                        "⬇️ Download Product Ranking CSV",
                        data=product_ranking.to_csv(
                            index=False
                        ).encode("utf-8"),
                        file_name="product_ranking.csv",
                        mime="text/csv"
        )
                
                
        
