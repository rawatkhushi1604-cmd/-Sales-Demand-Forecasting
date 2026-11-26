import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Sales Demand Forecasting System",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Sales Demand Forecasting System")
st.markdown(
    """
    **A Smart Business Analytics and Product-wise Demand Forecasting Platform**

    Analyze sales performance, understand product demand,
    compare machine learning models and forecast future sales.
    """
)
st.divider()

# ==========================================
# HELPER FUNCTIONS
# ==========================================
def create_features(dates, start_time=1):
    dates = pd.DatetimeIndex(pd.to_datetime(dates))
    time_values = np.arange(start_time, start_time + len(dates))
    return pd.DataFrame({
        "Time": time_values,
        "Month": dates.month,
        "Year": dates.year,
        "Month_Sin": np.sin(2 * np.pi * dates.month / 12),
        "Month_Cos": np.cos(2 * np.pi * dates.month / 12)
    })


def calculate_metrics(actual, predicted):
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    r2 = r2_score(actual, predicted) if len(actual) >= 3 else np.nan
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
    temp[date_column] = pd.to_datetime(temp[date_column], errors="coerce")
    temp[value_column] = pd.to_numeric(temp[value_column], errors="coerce")
    temp = temp.dropna()

    if temp.empty:
        return pd.DataFrame(columns=["Date", "Value"])

    temp["Month"] = temp[date_column].dt.to_period("M").dt.to_timestamp()
    monthly = temp.groupby("Month")[value_column].sum().sort_index()

    all_months = pd.date_range(
        start=monthly.index.min(),
        end=monthly.index.max(),
        freq="MS"
    )
    monthly = monthly.reindex(all_months, fill_value=0)

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
    if len(monthly_data) - test_size < 4:
        return None

    train_X, test_X = X.iloc[:-test_size], X.iloc[-test_size:]
    train_y, test_y = y[:-test_size], y[-test_size:]

    results = {}
    predictions = {}

    for name, model in get_models().items():
        model.fit(train_X, train_y)
        predicted = np.maximum(model.predict(test_X), 0)
        mae, rmse, r2 = calculate_metrics(test_y, predicted)
        results[name] = {
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2
        }
        predictions[name] = predicted

    best_model_name = min(results, key=lambda name: results[name]["MAE"])

    return {
        "results": results,
        "predictions": predictions,
        "best_model": best_model_name,
        "actual": test_y,
        "test_dates": monthly_data["Date"].iloc[-test_size:]
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

    future_X = create_features(future_dates, start_time=len(monthly_data) + 1)
    predictions = np.maximum(model.predict(future_X), 0)

    return pd.DataFrame({
        "Date": future_dates,
        "Predicted Value": predictions
    })


def format_date_col(df_in, col="Date"):
    out = df_in.copy()
    out[col] = pd.to_datetime(out[col]).dt.strftime("%Y-%m-%d")
    return out


# ==========================================
# DATASET UPLOAD
# ==========================================
st.header("📂 Upload Your Sales Dataset")
uploaded_file = st.file_uploader(
    "Upload your CSV file",
    type=["csv"],
    key="main_csv_uploader"
)

if uploaded_file is None:
    st.info("Please upload a CSV dataset to start Business Analytics and Demand Forecasting.")
    st.stop()

try:
    df = pd.read_csv(uploaded_file)
except Exception as e:
    st.error(f"Unable to read CSV file: {e}")
    st.stop()

if df.empty:
    st.error("The uploaded dataset is empty.")
    st.stop()

# ==========================================
# AUTOMATIC COLUMN DETECTION
# ==========================================
date_candidates = [col for col in df.columns if "date" in col.lower()]
numeric_candidates = [
    col for col in df.select_dtypes(include=np.number).columns
    if df[col].nunique() > 1
]
category_candidates = [
    col for col in df.columns
    if col not in numeric_candidates and col not in date_candidates
]


def get_default(options, preferred):
    return options.index(preferred) if preferred in options else 0


# ==========================================
# SIDEBAR SETTINGS
# ==========================================
# ==========================================
st.sidebar.header("⚙️ Business Analytics Settings")

date_options = ["Not selected"] + date_candidates
numeric_options = ["Not selected"] + numeric_candidates
category_options = ["Not selected"] + category_candidates

date_col = st.sidebar.selectbox(
    "Date Column",
    date_options,
    index=get_default(date_options, "Order Date")
)

numeric_col = st.sidebar.selectbox(
    "Sales / Revenue Column",
    numeric_options,
    index=get_default(numeric_options, "Sales")
)

category_col = st.sidebar.selectbox(
    "Category Column",
    category_options,
    index=get_default(category_options, "Category")
)

st.sidebar.header("📦 Product Forecasting Settings")

product_options = ["Not selected"] + category_candidates
product_col = st.sidebar.selectbox(
    "Product Name Column",
    product_options,
    index=get_default(product_options, "Product Name")
)

demand_options = ["Not selected"] + numeric_candidates
demand_col = st.sidebar.selectbox(
    "Product Demand Column",
    demand_options,
    index=get_default(demand_options, "Quantity")
)

forecast_periods = st.sidebar.selectbox(
    "Forecast Period (Months)",
    [3, 6, 9, 12],
    index=1
)

st.sidebar.success("Dataset loaded successfully!")
st.success("CSV Dataset Uploaded Successfully!")

# ==========================================
# DATASET PREVIEW
# ==========================================
st.subheader("👀 Dataset Preview")
st.dataframe(df.head(10), use_container_width=True)

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Records", f"{len(df):,}")
with col2:
    st.metric("Total Columns", len(df.columns))
with col3:
    st.metric(
        "Unique Products",
        f"{df[product_col].nunique():,}" if product_col != "Not selected" else "Not Selected"
    )

st.divider()

# ==========================================
# BUSINESS ANALYTICS
# ==========================================
st.header("📊 Business Analytics Dashboard")

if date_col == "Not selected" or numeric_col == "Not selected":
    st.warning("Please select a valid Date Column and Sales / Revenue Column.")
else:
    business_data = df.copy()
    business_data[date_col] = pd.to_datetime(business_data[date_col], errors="coerce")
    business_data[numeric_col] = pd.to_numeric(business_data[numeric_col], errors="coerce")
    business_data = business_data.dropna(subset=[date_col, numeric_col])
    business_data = business_data[business_data[numeric_col] >= 0]

    if business_data.empty:
        st.warning("No valid sales records found.")
    else:
        total_sales = business_data[numeric_col].sum()
        average_sales = business_data[numeric_col].mean()
        total_records = len(business_data)

        st.subheader("📌 Key Business Performance Indicators")
        k1, k2, k3 = st.columns(3)
        k1.metric("Total Revenue / Sales", f"₹{total_sales:,.2f}")
        k2.metric("Average Sales", f"₹{average_sales:,.2f}")
        k3.metric("Valid Sales Records", f"{total_records:,}")

        st.divider()

        # CATEGORY-WISE SALES ANALYSIS
        st.subheader("🏷️ Category-wise Sales Analysis")
        if category_col != "Not selected":
            category_data = (
                business_data.groupby(category_col)[numeric_col]
                .sum()
                .sort_values(ascending=False)
            )
            if not category_data.empty:
                st.bar_chart(category_data)
                cat_table = category_data.reset_index().rename(
                    columns={category_col: "Category", numeric_col: "Total Sales"}
                )
                cat_table["Total Sales"] = cat_table["Total Sales"].round(2)
                cat_table.index = np.arange(1, len(cat_table) + 1)
                st.dataframe(cat_table, use_container_width=True)
        else:
            st.info("Select a Category Column from sidebar to view category-wise analysis.")

        st.divider()

        # MONTHLY SALES TREND
        st.subheader("📅 Monthly Sales Trend")
        monthly_sales = make_monthly_data(business_data, date_col, numeric_col)

        if len(monthly_sales) >= 2:
            monthly_chart = monthly_sales.set_index("Date")
            st.line_chart(monthly_chart["Value"])

            monthly_table = format_date_col(monthly_sales, "Date").rename(
                columns={"Date": "Month", "Value": "Total Sales"}
            )
            monthly_table["Total Sales"] = monthly_table["Total Sales"].round(2)
            monthly_table.index = np.arange(1, len(monthly_table) + 1)
            st.dataframe(monthly_table, use_container_width=True)
        else:
            st.warning("At least two months of sales history are required for a monthly trend.")

        st.divider()

        # OVERALL SALES FORECASTING
        st.header("🔮 Overall Sales Forecasting")

        if len(monthly_sales) >= 6:
            evaluation = train_and_evaluate(monthly_sales)

            if evaluation is not None:
                st.subheader("🤖 Machine Learning Model Comparison")

                results_df = pd.DataFrame(evaluation["results"]).T.reset_index()
                results_df = results_df.rename(columns={"index": "Model"})
                results_df.index = np.arange(1, len(results_df) + 1)
                st.dataframe(results_df.round(3), use_container_width=True)

                best_model_name = evaluation["best_model"]
                st.success(
                    f"Best Performing Model: {best_model_name} (Selected using lowest MAE)"
                )

                # ACTUAL VS PREDICTED
                st.subheader("📈 Actual vs Predicted Sales")

                comparison_df = pd.DataFrame({
                    "Date": evaluation["test_dates"].values,
                    "Actual Sales": evaluation["actual"],
                    "Linear Regression": evaluation["predictions"]["Linear Regression"],
                    "Random Forest": evaluation["predictions"]["Random Forest"]
                })

                chart_comparison = comparison_df.set_index("Date")
                st.line_chart(
                    chart_comparison[["Actual Sales", "Linear Regression", "Random Forest"]]
                )

                comp_display = format_date_col(comparison_df, "Date").round(2)
                comp_display.index = np.arange(1, len(comp_display) + 1)
                st.dataframe(comp_display, use_container_width=True)

                st.divider()

                # FUTURE SALES FORECAST
                st.subheader(f"📆 Next {forecast_periods} Months Sales Forecast")

                future_forecast = forecast_future(
                    monthly_sales,
                    best_model_name,
                    forecast_periods
                )
                future_forecast["Predicted Value"] = future_forecast["Predicted Value"].round(2)

                st.line_chart(future_forecast.set_index("Date")["Predicted Value"])

                future_display = format_date_col(future_forecast, "Date")
                future_display.index = np.arange(1, len(future_display) + 1)
                st.dataframe(future_display, use_container_width=True)

                # DOWNLOADS
                forecast_csv = format_date_col(future_forecast, "Date").to_csv(
                    index=False
                ).encode("utf-8")
                st.download_button(
                                      data=forecast_csv,
                    file_name="overall_sales_forecast.csv",
                    mime="text/csv"
                )

                metrics_csv = results_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="⬇️ Download Model Performance Report",
                    data=metrics_csv,
                    file_name="model_performance_report.csv",
                    mime="text/csv"
                )
        else:
            st.warning(
                "At least 6 months of historical sales data are required for model training and forecasting."
            )

st.divider()

# ==========================================
# AUTOMATIC PRODUCT-WISE DEMAND FORECASTING
# ==========================================
st.header("📦 Automatic Product-wise Demand Forecasting")

if (
    date_col == "Not selected"
    or product_col == "Not selected"
    or demand_col == "Not selected"
):
    st.info(
        "Product-wise forecasting is not available for this dataset "
        "because Product Name or Quantity column is missing. "
        "Your Business Analytics and Overall Sales Forecasting "
        "sections can still work normally."
    )
else:
    product_data = df.copy()
    product_data[date_col] = pd.to_datetime(product_data[date_col], errors="coerce")
    product_data[demand_col] = pd.to_numeric(product_data[demand_col], errors="coerce")
    product_data = product_data.dropna(subset=[date_col, product_col, demand_col])
    product_data = product_data[product_data[demand_col] >= 0]
    product_data[product_col] = product_data[product_col].astype(str).str.strip()
    product_data = product_data[product_data[product_col] != ""]

    all_products = sorted(product_data[product_col].unique())
    st.success(f"Total Products Detected: {len(all_products)}")

    all_forecasts = []
    all_metrics = []
    skipped_products = []

    for product in all_products:
        product_history = product_data[product_data[product_col] == product].copy()
        monthly_product = make_monthly_data(product_history, date_col, demand_col)

        if len(monthly_product) < 6:
            skipped_products.append(product)
            continue

        evaluation = train_and_evaluate(monthly_product)
        if evaluation is None:
            skipped_products.append(product)
            continue

        best_model = evaluation["best_model"]
        future = forecast_future(monthly_product, best_model, forecast_periods)
        future["Product"] = product
        future = future.rename(columns={"Predicted Value": "Predicted Demand"})
        future["Predicted Demand"] = future["Predicted Demand"].round(0).astype(int)

        all_forecasts.append(future)

        model_result = evaluation["results"][best_model]
        all_metrics.append({
            "Product": product,
            "Best Model": best_model,
            "MAE": round(model_result["MAE"], 2),
            "RMSE": round(model_result["RMSE"], 2),
            "R2": round(model_result["R2"], 3) if pd.notna(model_result["R2"]) else np.nan
        })

    if all_forecasts:
        final_forecast = pd.concat(all_forecasts, ignore_index=True)
        final_forecast = final_forecast[["Product", "Date", "Predicted Demand"]]

        st.subheader("🔮 Forecast for All Products")
        forecast_display = format_date_col(final_forecast, "Date")
        forecast_display.index = np.arange(1, len(forecast_display) + 1)
        st.dataframe(forecast_display, use_container_width=True)

        st.subheader("🤖 Product-wise Model Performance")
        metrics_df = pd.DataFrame(all_metrics)
        metrics_df.index = np.arange(1, len(metrics_df) + 1)
        st.dataframe(metrics_df, use_container_width=True)

        st.subheader("📊 Product Demand Forecast Chart")
        chart_data = final_forecast.pivot(
            index="Date",
            columns="Product",
            values="Predicted Demand"
        )
        st.line_chart(chart_data)

        forecast_download = format_date_col(final_forecast, "Date").to_csv(
            index=False
        ).encode("utf-8")
        st.download_button(
            label="⬇️ Download All Product Forecasts",
            data=forecast_download,
            file_name="all_product_demand_forecast.csv",
            mime="text/csv"
        )

        metrics_download = metrics_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download Product Model Performance",
            data=metrics_download,
            file_name="product_model_performance.csv",
            mime="text/csv"
        )

        st.divider()

        # ==========================================
        # PRODUCT DEMAND RANKING
        # ==========================================
        st.header("🏆 Product Demand Ranking")

        historical_ranking = (
            product_data.groupby(product_col)[demand_col]
            .sum()
            .sort_values(ascending=False)
            .reset_index()
        )
        historical_ranking.columns = ["Product", "Total Historical Demand"]
        historical_ranking["Total Historical Demand"] = (
            historical_ranking["Total Historical Demand"].round(0).astype(int)
        )
        historical_ranking.insert(0, "Rank", range(1, len(historical_ranking) + 1))
        historical_ranking.index = np.arange(1, len(historical_ranking) + 1)

        st.subheader("🥇 Highest Selling Products")
        st.dataframe(historical_ranking, use_container_width=True)
        st.bar_chart(
            historical_ranking.head(10).set_index("Product")["Total Historical Demand"]
        )

        ranking_download = historical_ranking.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download Product Demand Ranking",
            data=ranking_download,
            file_name="product_demand_ranking.csv",
            mime="text/csv"
        )

        st.divider()
        # ==========================================
        # BUSINESS RECOMMENDATIONS
        # ==========================================
        st.header("💡 Smart Business Recommendations")

        average_product_demand = final_forecast.groupby("Product")["Predicted Demand"].mean()
        overall_average_demand = average_product_demand.mean()

        high_demand_products = average_product_demand[
            average_product_demand >= overall_average_demand
        ].sort_values(ascending=False)

        low_demand_products = average_product_demand[
            average_product_demand < overall_average_demand
        ].sort_values()

        recommendation_col1, recommendation_col2 = st.columns(2)

        with recommendation_col1:
            st.subheader("📈 High Demand Products")
            if not high_demand_products.empty:
                for product, demand in high_demand_products.items():
                    st.write(f"• **{product}** — Average forecast demand: {demand:.0f} units")
            else:
                st.info("No products above average forecast demand.")

        with recommendation_col2:
            st.subheader("📉 Lower Demand Products")
            if not low_demand_products.empty:
                for product, demand in low_demand_products.items():
                    st.write(f"• **{product}** — Average forecast demand: {demand:.0f} units")
            else:
                st.info("No products below average forecast demand.")

        st.subheader("🧠 Business Insights")
        st.write("1. Products with higher forecast demand may need greater attention during purchasing and replenishment planning.")
        st.write("2. Products with lower forecast demand can be reviewed for pricing, promotion and marketing opportunities.")
        st.write("3. Use product-wise model performance to understand how accurately demand is being predicted.")
        st.write("4. Compare forecasts with actual inventory, supplier lead time and business requirements before making stock decisions.")  label="⬇️ Download Overall Sales Forecast",
                
