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

    r2 = (
        r2_score(actual, predicted)
        if len(actual) >= 2
        else np.nan
    )

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
        return pd.DataFrame(
            columns=["Date", "Value"]
        )

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

    test_size = min(
        6,
        max(2, len(monthly_data) // 5)
    )

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
    
# DATASET UPLOAD

st.header("📂 Upload Your Sales Dataset")

uploaded_file = st.file_uploader(
    "Choose your CSV file",
    type=["csv"],
    key="main_csv_uploader"
)

if uploaded_file is None:
    st.info("Please upload your CSV file to start forecasting.")
    st.stop()

try:
    df = pd.read_csv(uploaded_file)
except Exception as e:
    st.error(f"Unable to read CSV file: {e}")
    st.stop()

if df.empty:
    st.error("The uploaded dataset is empty.")
    st.stop()


# AUTOMATIC COLUMN DETECTION

date_candidates = [
    col for col in df.columns
    if "date" in col.lower()
]

numeric_candidates = [
    col for col in df.select_dtypes(include=np.number).columns
    if df[col].nunique() > 1
]

category_candidates = [
    col for col in df.columns
    if col not in numeric_candidates
    and col not in date_candidates
]

# DEFAULT COLUMN SELECTION

def get_default(options, preferred):
    if preferred in options:
        return options.index(preferred)
    return 0

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
 
# BUSINESS ANALYTICS

st.header("📈 Business Analytics")

if date_col == "Not selected" or numeric_col == "Not selected":

    st.warning(
        "Please select both Date Column and Sales / Revenue Column "
        "from the sidebar to perform Business Analytics."
    )

else:

    business_data = df.copy()

    business_data[date_col] = pd.to_datetime(
        business_data[date_col],
        errors="coerce"
    )

    business_data[numeric_col] = pd.to_numeric(
        business_data[numeric_col],
        errors="coerce"
    )

    business_data = business_data.dropna(
        subset=[date_col, numeric_col]
    )

    business_data = business_data[
        business_data[numeric_col] >= 0
    ]

    if business_data.empty:

        st.error("No valid sales data found in the selected columns.")

    else:

        total_sales = business_data[numeric_col].sum()
        average_sales = business_data[numeric_col].mean()
        total_transactions = len(business_data)

        metric1, metric2, metric3 = st.columns(3)

        metric1.metric(
            "Total Revenue / Sales",
            f"{total_sales:,.2f}"
        )

        metric2.metric(
            "Average Sales",
            f"{average_sales:,.2f}"
        )

        metric3.metric(
            "Total Records",
            f"{total_transactions:,}"
        )

        # CATEGORY-WISE ANALYSIS

        if category_col != "Not selected":

            st.subheader("🏷️ Category-wise Sales Analysis")

            category_sales = (
                business_data.groupby(category_col)[numeric_col]
                .sum()
                .sort_values(ascending=False)
            )

            st.dataframe(
                category_sales.reset_index(),
                use_container_width=True
            )

            fig, ax = plt.subplots(figsize=(10, 5))

            category_sales.plot(
                kind="bar",
                ax=ax
            )

            ax.set_title("Category-wise Sales")
            ax.set_xlabel("Category")
            ax.set_ylabel("Sales / Revenue")
            plt.xticks(rotation=45)
            plt.tight_layout()

            st.pyplot(fig)
            plt.close(fig)

        # MONTHLY SALES ANALYSIS

        st.subheader("📅 Monthly Sales Analysis")

        monthly_data = make_monthly_data(
            business_data,
            date_col,
            numeric_col
        )

        if len(monthly_data) >= 2:

            fig, ax = plt.subplots(figsize=(12, 5))

            ax.plot(
                monthly_data["Date"],
                monthly_data["Value"],
                marker="o"
            )

            ax.set_title("Historical Monthly Sales")
            ax.set_xlabel("Month")
            ax.set_ylabel("Sales / Revenue")
            ax.grid(True)
            plt.xticks(rotation=45)
            plt.tight_layout()

            st.pyplot(fig)
            plt.close(fig)

            # OVERALL SALES FORECAST

            st.header("🔮 Overall Sales Forecast")

            if len(monthly_data) < 6:

                st.warning(
                    "At least 6 months of historical data is required "
                    "for forecasting."
                )

            else:

                evaluation = train_and_evaluate(monthly_data)

                if evaluation is not None:

                    st.subheader("🤖 Model Performance Comparison")

                    performance_data = pd.DataFrame(
                        evaluation["results"]
                    ).T

                    st.dataframe(
                        performance_data,
                        use_container_width=True
                    )

                    best_model_name = evaluation["best_model"]

                    st.success(
                        f"Best Performing Model: {best_model_name}"
                    )

                    st.subheader("📊 Actual vs Predicted Sales")

                    comparison = pd.DataFrame({
                        "Date": evaluation["test_dates"].values,
                        "Actual Sales": evaluation["actual"],
                        "Predicted Sales": evaluation[
                            "predictions"
                        ][best_model_name]
                    })

                    st.dataframe(
                        comparison,
                        use_container_width=True
                    )

                    fig, ax = plt.subplots(figsize=(12, 5))

                    ax.plot(
                        comparison["Date"],
                        comparison["Actual Sales"],
                        marker="o",
                        label="Actual Sales"
                    )

                    ax.plot(
                        comparison["Date"],
                        comparison["Predicted Sales"],
                        marker="o",
                        label="Predicted Sales"
                    )

                    ax.set_title("Actual vs Predicted Sales")
                    ax.set_xlabel("Month")
                    ax.set_ylabel("Sales / Revenue")
                    ax.legend()
                    ax.grid(True)
                    plt.xticks(rotation=45)
                    plt.tight_layout()

                    st.pyplot(fig)
                    plt.close(fig)

                    # FUTURE FORECAST

                    future_forecast = forecast_future(
                        monthly_data,
                        best_model_name,
                        forecast_periods
                    )

                    st.subheader(
                        f"📅 Next {forecast_periods} Months Sales Forecast"
                    )

                    st.dataframe(
                        future_forecast,
                        use_container_width=True
                    )

                    fig, ax = plt.subplots(figsize=(12, 5))

                    ax.plot(
                        monthly_data["Date"],
                        monthly_data["Value"],
                        marker="o",
                        label="Historical Sales"
                    )

                    ax.plot(
                        future_forecast["Date"],
                        future_forecast["Predicted Value"],
                        marker="o",
                        linestyle="--",
                        label="Forecasted Sales"
                    )

                    ax.set_title("Future Sales Forecast")
                    ax.set_xlabel("Month")
                    ax.set_ylabel("Sales / Revenue")
                    ax.legend()
                    ax.grid(True)
                    plt.xticks(rotation=45)
                    plt.tight_layout()

                    st.pyplot(fig)
                    plt.close(fig)

                    csv_data = future_forecast.to_csv(
                        index=False
                    ).encode("utf-8")

                    st.download_button(
                        "⬇️ Download Overall Sales Forecast",
                        data=csv_data,
                        file_name="overall_sales_forecast.csv",
                        mime="text/csv"
                    )

        else:

            st.warning(
                "Not enough monthly data is available for analysis."
)
# PRODUCT-WISE DEMAND FORECASTING

st.header("📦 Product-wise Demand Forecasting")

if (
    date_col == "Not selected"
    or product_col == "Not selected"
    or demand_col == "Not selected"
):

    st.warning(
        "Please select Date Column, Product Name Column "
        "and Product Demand Column from the sidebar."
    )

else:

    product_data = df.copy()

    product_data[date_col] = pd.to_datetime(
        product_data[date_col],
        errors="coerce"
    )

    product_data[demand_col] = pd.to_numeric(
        product_data[demand_col],
        errors="coerce"
    )

    product_data = product_data.dropna(
        subset=[date_col, product_col, demand_col]
    )

    product_data = product_data[
        product_data[demand_col] >= 0
    ]

    if product_data.empty:

        st.error("No valid product demand data found.")

    else:

        available_products = sorted(
            product_data[product_col].astype(str).unique()
        )

        selected_product = st.selectbox(
            "Select a Product",
            available_products
        )

        product_history = product_data[
            product_data[product_col].astype(str)
            == selected_product
        ].copy()

        monthly_product = make_monthly_data(
            product_history,
            date_col,
            demand_col
        )

        st.subheader(
            f"📊 Historical Demand: {selected_product}"
        )

        if not monthly_product.empty:

            fig, ax = plt.subplots(figsize=(12, 5))

            ax.plot(
                monthly_product["Date"],
                monthly_product["Value"],
                marker="o"
            )

            ax.set_title(
                f"Monthly Demand of {selected_product}"
            )

            ax.set_xlabel("Month")
            ax.set_ylabel("Units Sold / Demand")
            ax.grid(True)
            plt.xticks(rotation=45)
            plt.tight_layout()

            st.pyplot(fig)
            plt.close(fig)

        if len(monthly_product) < 6:

            st.warning(
                "At least 6 months of product history is required "
                "for demand forecasting."
            )

        else:

            product_evaluation = train_and_evaluate(
                monthly_product
            )

            if product_evaluation is not None:

                st.subheader("🤖 Product Model Comparison")

                product_performance = pd.DataFrame(
                    product_evaluation["results"]
                ).T

                st.dataframe(
                    product_performance,
                    use_container_width=True
                )

                product_best_model = product_evaluation[
                    "best_model"
                ]

                st.success(
                    f"Best Model for {selected_product}: "
                    f"{product_best_model}"
                )

                st.subheader("📉 Product Forecast Accuracy")

                product_comparison = pd.DataFrame({
                    "Date": product_evaluation["test_dates"].values,
                    "Actual Demand": product_evaluation["actual"],
                    "Predicted Demand": product_evaluation[
                        "predictions"
                    ][product_best_model]
                })

                st.dataframe(
                    product_comparison,
                    use_container_width=True
                )

                fig, ax = plt.subplots(figsize=(12, 5))

                ax.plot(
                    product_comparison["Date"],
                    product_comparison["Actual Demand"],
                    marker="o",
                    label="Actual Demand"
                )

                ax.plot(
                    product_comparison["Date"],
                    product_comparison["Predicted Demand"],
                    marker="o",
                    label="Predicted Demand"
                )

                ax.set_title(
                    f"Actual vs Predicted Demand: {selected_product}"
                )

                ax.set_xlabel("Month")
                ax.set_ylabel("Demand (Units)")
                ax.legend()
                ax.grid(True)
                plt.xticks(rotation=45)
                plt.tight_layout()

                st.pyplot(fig)
                plt.close(fig)

                # FUTURE PRODUCT DEMAND

                product_future = forecast_future(
                    monthly_product,
                    product_best_model,
                    forecast_periods
                )

                product_future = product_future.rename(
                    columns={
                        "Predicted Value": "Predicted Demand"
                    }
                )

                product_future["Predicted Demand"] = (
                    product_future["Predicted Demand"].round(0)
                )

                st.subheader(
                    f"🔮 Next {forecast_periods} Months Demand Forecast"
                )

                st.dataframe(
                    product_future,
                    use_container_width=True
                )

                fig, ax = plt.subplots(figsize=(12, 5))

                ax.plot(
                    monthly_product["Date"],
                    monthly_product["Value"],
                    marker="o",
                    label="Historical Demand"
                )

                ax.plot(
                    product_future["Date"],
                    product_future["Predicted Demand"],
                    marker="o",
                    linestyle="--",
                    label="Future Demand"
                )

                ax.set_title(
                    f"Future Demand Forecast: {selected_product}"
                )

                ax.set_xlabel("Month")
                ax.set_ylabel("Demand (Units)")
                ax.legend()
                ax.grid(True)
                plt.xticks(rotation=45)
                plt.tight_layout()

                st.pyplot(fig)
                plt.close(fig)

                product_csv = product_future.to_csv(
                    index=False
                ).encode("utf-8")

                st.download_button(
                    "⬇️ Download Product Demand Forecast",
                    data=product_csv,
                    file_name="product_demand_forecast.csv",
                    mime="text/csv"
)
        
# ALL PRODUCTS DEMAND COMPARISON

st.header("🏆 Product Demand Comparison")

if (
    product_col != "Not selected"
    and demand_col != "Not selected"
):

    comparison_data = df.copy()

    comparison_data[demand_col] = pd.to_numeric(
        comparison_data[demand_col],
        errors="coerce"
    )

    comparison_data = comparison_data.dropna(
        subset=[product_col, demand_col]
    )

    comparison_data = comparison_data[
        comparison_data[demand_col] >= 0
    ]

    if not comparison_data.empty:

        product_summary = (
            comparison_data.groupby(product_col)[demand_col]
            .agg(
                Total_Demand="sum",
                Average_Demand="mean",
                Total_Orders="count"
            )
            .sort_values(
                by="Total_Demand",
                ascending=False
            )
            .reset_index()
        )

        product_summary["Total_Demand"] = (
            product_summary["Total_Demand"].round(2)
        )

        product_summary["Average_Demand"] = (
            product_summary["Average_Demand"].round(2)
        )

        st.subheader("📋 Product Performance Ranking")

        st.dataframe(
            product_summary,
            use_container_width=True
        )

        fig, ax = plt.subplots(figsize=(12, 6))

        ax.bar(
            product_summary[product_col].astype(str),
            product_summary["Total_Demand"]
        )

        ax.set_title("Total Demand by Product")
        ax.set_xlabel("Product")
        ax.set_ylabel("Total Units Sold")
        plt.xticks(rotation=60, ha="right")
        plt.tight_layout()

        st.pyplot(fig)
        plt.close(fig)

        best_product = product_summary.iloc[0]

        st.success(
            f"Highest Demand Product: "
            f"{best_product[product_col]} "
            f"({best_product['Total_Demand']:,.0f} units)"
        )

        summary_csv = product_summary.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            "⬇️ Download Product Performance Report",
            data=summary_csv,
            file_name="product_performance_report.csv",
            mime="text/csv"
        )

    else:

        st.warning("No valid product data available.")

else:

    st.info(
        "Select Product Name Column and Product Demand Column "
        "to view product comparison."
    )


# PROJECT INFORMATION

st.divider()

st.subheader("ℹ️ About This Project")

st.write(
    """
    **Sales Demand Forecasting System**

    This application provides:

    - Business Analytics and Revenue Analysis
    - Category-wise Sales Analysis
    - Monthly Sales Trends
    - Overall Sales Forecasting
    - Product-wise Demand Forecasting
    - Linear Regression and Random Forest Models
    - Model Performance Evaluation using MAE, RMSE and R²
    - Downloadable Forecast Reports

    The application supports flexible CSV datasets containing
    date, sales, product and demand information.
    """
)

st.caption(
    "Developed as a B.Tech Computer Science and Data Science Project."
)
