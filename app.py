
import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# Page Configuration
st.set_page_config(
    page_title="Flexible Business Analytics",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Flexible Business Analytics & Forecasting System")
st.write(
    "Upload any CSV dataset and explore its business insights."
)

# Upload Dataset
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

    # Dataset Overview
    st.subheader("1. Dataset Overview")

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Records", len(data))
    col2.metric("Total Columns", len(data.columns))
    col3.metric(
        "Missing Values",
        int(data.isnull().sum().sum())
    )

    st.write("Dataset Preview")
    st.dataframe(data.head(10), use_container_width=True)

    # Automatic Column Detection
    st.subheader("2. Select Columns for Analysis")

    date_names = [
        "date", "order date", "transaction date",
        "purchase date", "invoice date"
    ]

    metric_names = [
        "sales", "revenue", "amount",
        "profit", "quantity", "total"
    ]

    date_candidates = [
        col for col in data.columns
        if str(col).strip().lower() in date_names
        or "date" in str(col).strip().lower()
    ]

    numeric_columns = []

    for col in data.columns:
        converted = pd.to_numeric(
            data[col], errors="coerce"
        )
        if converted.notna().sum() > 0:
            numeric_columns.append(col)

    category_columns = [
        col for col in data.columns
        if col not in numeric_columns
        and col not in date_candidates
    ]

    # Date Selection
    date_options = ["Not selected"] + list(data.columns)

    date_default = (
        date_options.index(date_candidates[0])
        if date_candidates else 0
    )

    date_col = st.selectbox(
        "Select Date Column (Optional)",
        date_options,
        index=date_default
    )

    # Numeric Column Selection
    metric_default = 0

    for i, col in enumerate(numeric_columns):
        if str(col).strip().lower() in metric_names:
            metric_default = i
            break

    metric_col = "Not selected"

    if numeric_columns:
        metric_col = st.selectbox(
            "Select Numeric Column to Analyze",
            numeric_columns,
            index=metric_default
        )
    else:
        st.warning("No numeric columns detected.")

    # Category Selection
    category_options = ["Not selected"] + category_columns

    category_col = st.selectbox(
        "Select Category Column (Optional)",
        category_options
    )

    # Business Insights
    st.subheader("3. Business Insights")

    if metric_col != "Not selected":

        values = pd.to_numeric(
            data[metric_col], errors="coerce"
        ).dropna()

        if not values.empty:

            col1, col2, col3 = st.columns(3)

            col1.metric("Total Value", f"{values.sum():,.2f}")
            col2.metric("Average Value", f"{values.mean():,.2f}")
            col3.metric("Maximum Value", f"{values.max():,.2f}")

            st.write("Minimum Value:", f"{values.min():,.2f}")

            st.subheader(f"{metric_col} Distribution")
            st.bar_chart(values.head(20))

            # Category Analysis
            if category_col != "Not selected":

                st.subheader("Category-wise Analysis")

                category_data = pd.DataFrame({
                    "Category": data[category_col],
                    "Value": pd.to_numeric(
                        data[metric_col], errors="coerce"
                    )
                }).dropna()

                category_summary = (
                    category_data.groupby("Category")["Value"]
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

    # Forecasting
    st.subheader("4. Demand Forecasting")

    if date_col != "Not selected" and metric_col != "Not selected":

        forecast_data = pd.DataFrame({
            "Date": pd.to_datetime(
                data[date_col], errors="coerce"
            ),
            "Value": pd.to_numeric(
                data[metric_col], errors="coerce"
            )
        }).dropna()

        if not forecast_data.empty:

            monthly = (
                forecast_data.set_index("Date")["Value"]
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
                    1, len(monthly) + 1
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
                    test["Value"], predictions
                )

                rmse = mean_squared_error(
                    test["Value"], predictions
                ) ** 0.5

                r2 = r2_score(
                    test["Value"], predictions
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
                    1, 12, 6
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
                    "Download Forecast CSV",
                    forecast.to_csv(index=False),
                    "Future_Forecast.csv",
                    "text/csv"
                )

            else:
                st.info(
                    "At least 12 monthly periods are needed for forecasting."
                )

        else:
            st.warning("No valid date and numeric values found.")

    else:
        st.info(
            "Select both a Date Column and a Numeric Column to enable forecasting."
        )

else:
    st.info("Please upload a CSV file to begin analysis.")
    
