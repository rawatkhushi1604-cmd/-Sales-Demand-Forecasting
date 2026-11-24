
import streamlit as st
import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# -----------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------

st.set_page_config(
    page_title="Business Analytics & Product Forecasting",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Business Analytics & Demand Forecasting System")

st.write(
    "Upload any CSV dataset to explore business insights "
    "and forecast future demand for individual products."
)

# -----------------------------------------
# DATASET UPLOAD
# -----------------------------------------

uploaded_file = st.file_uploader(
    "Upload Any CSV Dataset",
    type=["csv"]
)

if uploaded_file is not None:

    try:
        data = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Unable to read CSV: {e}")
        st.stop()

    if data.empty:
        st.error("The uploaded CSV file is empty.")
        st.stop()

    # -----------------------------------------
    # DATASET OVERVIEW
    # -----------------------------------------

    st.header("1. Dataset Overview")

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Records", len(data))
    col2.metric("Total Columns", len(data.columns))
    col3.metric(
        "Missing Values",
        int(data.isnull().sum().sum())
    )

    st.write("Dataset Preview")

    st.dataframe(
        data.head(10),
        use_container_width=True
    )

    # -----------------------------------------
    # COLUMN DETECTION
    # -----------------------------------------

    st.header("2. Select Columns for Analysis")

    date_names = [
        "date",
        "order date",
        "transaction date",
        "purchase date",
        "invoice date",
        "month"
    ]

    metric_names = [
        "sales",
        "revenue",
        "amount",
        "profit",
        "quantity",
        "total"
    ]

    date_candidates = [
        col for col in data.columns
        if str(col).strip().lower() in date_names
        or "date" in str(col).strip().lower()
    ]

    numeric_columns = []

    for col in data.columns:

        converted = pd.to_numeric(
            data[col],
            errors="coerce"
        )

        if converted.notna().sum() > 0:
            numeric_columns.append(col)

    # -----------------------------------------
    # DATE SELECTION
    # -----------------------------------------

    date_options = ["Not selected"] + list(data.columns)

    date_default = (
        date_options.index(date_candidates[0])
        if date_candidates else 0
    )

    date_col = st.selectbox(
        "Select Date Column",
        date_options,
        index=date_default,
        key="main_date"
    )

    # -----------------------------------------
    # NUMERIC COLUMN SELECTION
    # -----------------------------------------

    metric_col = "Not selected"

    if numeric_columns:

        metric_default = 0

        for i, col in enumerate(numeric_columns):

            if str(col).strip().lower() in metric_names:
                metric_default = i
                break

        metric_col = st.selectbox(
            "Select Numeric Column",
            numeric_columns,
            index=metric_default,
            key="main_metric"
        )

    else:
        st.warning("No numeric columns detected.")

    # -----------------------------------------
    # CATEGORY SELECTION
    # -----------------------------------------

    category_columns = [
        col for col in data.columns
        if col not in numeric_columns
        and col != date_col
    ]

    category_options = ["Not selected"] + category_columns

    category_col = st.selectbox(
        "Select Category Column (Optional)",
        category_options,
        key="main_category"
    )

    # -----------------------------------------
    # BUSINESS ANALYTICS
    # -----------------------------------------

    st.header("3. Complete Business Analytics")

    if metric_col != "Not selected":

        values = pd.to_numeric(
            data[metric_col],
            errors="coerce"
        ).dropna()

        if not values.empty:

            col1, col2, col3, col4 = st.columns(4)

            col1.metric(
                "Total Value",
                f"{values.sum():,.2f}"
            )

            col2.metric(
                "Average Value",
                f"{values.mean():,.2f}"
            )

            col3.metric(
                "Maximum Value",
                f"{values.max():,.2f}"
            )

            col4.metric(
                "Minimum Value",
                f"{values.min():,.2f}"
            )

            # Distribution

            st.subheader(
                f"{metric_col} Distribution"
            )

            st.bar_chart(values.head(20))

            # Category Analysis

            if category_col != "Not selected":

                st.subheader("Category-wise Analysis")

                category_data = pd.DataFrame({
                    "Category": data[category_col],
                    "Value": pd.to_numeric(
                        data[metric_col],
                        errors="coerce"
                    )
                }).dropna()

                category_summary = (
                    category_data
                    .groupby("Category")["Value"]
                    .sum()
                    .sort_values(ascending=False)
                    .head(10)
                )

                st.bar_chart(category_summary)

                st.write("Category Summary")

                st.dataframe(
                    category_summary.reset_index(),
                    use_container_width=True
                )

    # -----------------------------------------
    # OVERALL SALES FORECASTING
    # -----------------------------------------

    st.header("4. Overall Demand Forecasting")

    if (
        date_col != "Not selected"
        and metric_col != "Not selected"
    ):

        forecast_data = pd.DataFrame({
            "Date": pd.to_datetime(
                data[date_col],
                errors="coerce"
            ),
            "Value": pd.to_numeric(
                data[metric_col],
                errors="coerce"
            )
        }).dropna()

        if not forecast_data.empty:

            monthly = (
                forecast_data
                .set_index("Date")["Value"]
                .resample("MS")
                .sum()
                .reset_index()
            )

            st.subheader("Monthly Trend")

            st.line_chart(
                monthly.set_index("Date")["Value"]
            )

            if len(monthly) >= 12:

                monthly["Time"] = range(
                    1,
                    len(monthly) + 1
                )

                train = monthly.iloc[:-6]
                test = monthly.iloc[-6:]

                model = LinearRegression()

                model.fit(
                    train[["Time"]],
                    train["Value"]
                )

                predictions = model.predict(
                    test[["Time"]]
                )

                mae = mean_absolute_error(
                    test["Value"],
                    predictions
                )

                rmse = np.sqrt(
                    mean_squared_error(
                        test["Value"],
                        predictions
                    )
                )

                r2 = r2_score(
                    test["Value"],
                    predictions
                )

                st.subheader("Model Evaluation")

                col1, col2, col3 = st.columns(3)

                col1.metric("MAE", f"{mae:,.2f}")
                col2.metric("RMSE", f"{rmse:,.2f}")
                col3.metric("R² Score", f"{r2:.2f}")

                st.subheader("Actual vs Predicted")

                comparison = pd.DataFrame({
                    "Actual": test["Value"].values,
                    "Predicted": predictions
                }, index=test["Date"])

                st.line_chart(comparison)

                # Future Forecast

                st.subheader("Future Forecast")

                periods = st.slider(
                    "Select Future Months",
                    1,
                    12,
                    6,
                    key="overall_periods"
                )

                final_model = LinearRegression()

                final_model.fit(
                    monthly[["Time"]],
                    monthly["Value"]
                )

                future_time = pd.DataFrame({
                    "Time": range(
                        len(monthly) + 1,
                        len(monthly) + periods + 1
                    )
                })

                future_dates = pd.date_range(
                    monthly["Date"].max()
                    + pd.offsets.MonthBegin(1),
                    periods=periods,
                    freq="MS"
                )

                forecast = pd.DataFrame({
                    "Date": future_dates,
                    "Predicted Value": final_model.predict(
                        future_time
                    ).round(2)
                })

                st.dataframe(
                    forecast,
                    use_container_width=True
                )

                st.line_chart(
                    forecast.set_index("Date")
                )

                st.download_button(
                    "Download Overall Forecast CSV",
                    forecast.to_csv(index=False),
                    "Overall_Forecast.csv",
                    "text/csv"
                )

            else:

                st.info(
                    "At least 12 monthly periods are needed "
                    "for overall forecasting."
                )

        else:

            st.warning(
                "No valid date and numeric values found."
            )

    else:

        st.info(
            "Select a Date Column and Numeric Column "
            "to enable overall forecasting."
        )

    # -----------------------------------------
    # PRODUCT-WISE DEMAND FORECASTING
    # -----------------------------------------

    st.header("5. Flexible Product-wise Demand Forecasting")

    st.write(
        "Forecast demand for individual products across "
        "different industries using your own dataset."
    )

    if (
        date_col != "Not selected"
        and numeric_columns
    ):

        product_columns = [
            col for col in data.columns
            if col != date_col
        ]

        col1, col2 = st.columns(2)

        with col1:

            product_col = st.selectbox(
                "Select Product Name Column",
                product_columns,
                key="product_name_column"
            )

        with col2:

            product_metric = st.selectbox(
                "Select Demand Column (Quantity, Sales, Revenue, etc.)",
                numeric_columns,
                key="product_metric_column"
            )

        product_periods = st.slider(
            "Select Product Forecast Months",
            1,
            12,
            6,
            key="product_periods"
        )

        if st.button(
            "Generate Product-wise Forecast",
            type="primary"
        ):

            product_data = pd.DataFrame({
                "Date": pd.to_datetime(
                    data[date_col],
                    errors="coerce"
                ),
                "Product": data[product_col],
                "Value": pd.to_numeric(
                    data[product_metric],
                    errors="coerce"
                )
            }).dropna()

            if product_data.empty:

                st.error(
                    "No valid product, date and numeric data found."
                )

            else:

                product_data["Product"] = (
                    product_data["Product"].astype(str)
                )

                latest_month = (
                    product_data["Date"].max()
                    .to_period("M")
                    .to_timestamp()
                )

                forecast_results = []
                evaluation_actual = []
                evaluation_predicted = []
                skipped_products = []

                # Forecast each product separately

                for product in product_data["Product"].unique():

                    one_product = product_data[
                        product_data["Product"] == product
                    ]

                    monthly_product = (
                        one_product
                        .set_index("Date")["Value"]
                        .resample("MS")
                        .sum()
                    )

                    full_months = pd.date_range(
                        monthly_product.index.min(),
                        latest_month,
                        freq="MS"
                    )

                    monthly_product = (
                        monthly_product
                        .reindex(full_months, fill_value=0)
                    )

                    monthly_product = (
                        monthly_product.reset_index(drop=True)
                    )

                    # Minimum historical data requirement

                    if len(monthly_product) < 12:

                        skipped_products.append(product)
                        continue

                    product_monthly = pd.DataFrame({
                        "Time": range(
                            1,
                            len(monthly_product) + 1
                        ),
                        "Value": monthly_product.values
                    })

                    # Train and test

                    train_product = product_monthly.iloc[:-6]
                    test_product = product_monthly.iloc[-6:]

                    product_model = LinearRegression()

                    product_model.fit(
                        train_product[["Time"]],
                        train_product["Value"]
                    )

                    test_predictions = product_model.predict(
                        test_product[["Time"]]
                    )

                    evaluation_actual.extend(
                        test_product["Value"].tolist()
                    )

                    evaluation_predicted.extend(
                        test_predictions.tolist()
                    )

                    # Train final model on all history

                    final_product_model = LinearRegression()

                    final_product_model.fit(
                        product_monthly[["Time"]],
                        product_monthly["Value"]
                    )

                    future_time = pd.DataFrame({
                        "Time": range(
                            len(product_monthly) + 1,
                            len(product_monthly)
                            + product_periods + 1
                        )
                    })

                    future_predictions = (
                        final_product_model
                        .predict(future_time)
                    )

                    future_predictions = np.maximum(
                        future_predictions,
                        0
                    )

                    future_dates = pd.date_range(
                        latest_month
                        + pd.offsets.MonthBegin(1),
                        periods=product_periods,
                        freq="MS"
                    )

                    for date, prediction in zip(
                        future_dates,
                        future_predictions
                    ):

                        forecast_results.append({
                            "Product": product,
                            "Month": date.strftime("%Y-%m"),
                            "Predicted Demand": round(
                                float(prediction), 2
                            )
                        })

                # -----------------------------------------
                # DISPLAY PRODUCT FORECASTS
                # -----------------------------------------

                if forecast_results:

                    result_df = pd.DataFrame(
                        forecast_results
                    )

                    st.success(
                        "Product-wise forecasting completed!"
                    )

                    st.subheader(
                        "Product-wise Forecast Results"
                    )

                    st.dataframe(
                        result_df,
                        use_container_width=True
                    )

                    # Product ranking

                    product_summary = (
                        result_df
                        .groupby("Product")["Predicted Demand"]
                        .sum()
                        .reset_index()
                        .rename(columns={
                            "Predicted Demand":
                            "Total Predicted Demand"
                        })
                        .sort_values(
                            "Total Predicted Demand",
                            ascending=False
                        )
                    )

                    st.subheader(
                        "Product Demand Ranking"
                    )

                    st.dataframe(
                        product_summary,
                        use_container_width=True
                    )

                    st.subheader(
                        "Predicted Demand by Product"
                    )

                    chart_data = result_df.pivot(
                        index="Month",
                        columns="Product",
                        values="Predicted Demand"
                    )

                    st.line_chart(chart_data)

                    # Highest predicted demand

                    top_product = product_summary.iloc[0]

                    st.subheader(
                        "Top Predicted Product"
                    )

                    st.metric(
                        "Highest Forecasted Demand",
                        str(top_product["Product"]),
                        f"{top_product['Total Predicted Demand']:,.2f} "
                        "total predicted units/value"
                    )

                    # Product model evaluation

                    if evaluation_actual:

                        st.subheader(
                            "Product Model Evaluation"
                        )

                        product_mae = mean_absolute_error(
                            evaluation_actual,
                            evaluation_predicted
                        )

                        product_rmse = np.sqrt(
                            mean_squared_error(
                                evaluation_actual,
                                evaluation_predicted
                            )
                        )

                        col1, col2 = st.columns(2)

                        col1.metric(
                            "Product MAE",
                            f"{product_mae:,.2f}"
                        )

                        col2.metric(
                            "Product RMSE",
                            f"{product_rmse:,.2f}"
                        )

                    # Download results

                    
            st.download_button(
                label="Download Product-wise Forecast CSV",
                data=result_df.to_csv(index=False),
                file_name="Product_Wise_Forecast.csv",
                mime="text/csv"
            )

