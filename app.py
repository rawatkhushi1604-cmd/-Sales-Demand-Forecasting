
import streamlit as st
import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(
    page_title="Sales Demand Forecasting",
    page_icon="📊",
    layout="wide"
)

st.title("Sales Demand Forecasting System")
st.write("Analyze sales data and predict future demand.")

uploaded_file = st.file_uploader(
    "Upload Sales CSV File",
    type=["csv"]
)

if uploaded_file is not None:

    try:
        data = pd.read_csv(uploaded_file)

        st.subheader("CSV Preview")
        st.dataframe(data.head())

        date_col = st.selectbox(
            "Select Date Column",
            data.columns
        )

        sales_col = st.selectbox(
            "Select Sales / Revenue Column",
            data.columns
        )

        if date_col == sales_col:
            st.error("Select two different columns.")

        else:
            data[date_col] = pd.to_datetime(
                data[date_col],
                errors="coerce"
            )

            data[sales_col] = pd.to_numeric(
                data[sales_col],
                errors="coerce"
            )

            data = data.dropna(
                subset=[date_col, sales_col]
            )

            if len(data) < 12:
                st.error(
                    "Please upload at least 12 valid records."
                )

            else:
                monthly = (
                    data.set_index(date_col)[sales_col]
                    .resample("ME")
                    .sum()
                    .reset_index()
                )

                monthly.columns = ["Date", "Sales"]

                monthly["Time"] = range(
                    1, len(monthly) + 1
                )

                if len(monthly) < 12:
                    st.error(
                        "Please upload data covering at least 12 months."
                    )

                else:

                    st.subheader("Sales Overview")

                    col1, col2 = st.columns(2)

                    col1.metric(
                        "Total Sales",
                        round(monthly["Sales"].sum(), 2)
                    )

                    col2.metric(
                        "Total Months",
                        len(monthly)
                    )

                    st.subheader("Monthly Sales Trend")

                    st.line_chart(
                        monthly.set_index("Date")["Sales"]
                    )

                    # MODEL EVALUATION

                    train = monthly.iloc[:-6]
                    test = monthly.iloc[-6:]

                    evaluation_model = LinearRegression()

                    evaluation_model.fit(
                        train[["Time"]],
                        train["Sales"]
                    )

                    test_predictions = evaluation_model.predict(
                        test[["Time"]]
                    )

                    mae = mean_absolute_error(
                        test["Sales"],
                        test_predictions
                    )

                    rmse = np.sqrt(
                        mean_squared_error(
                            test["Sales"],
                            test_predictions
                        )
                    )

                    r2 = r2_score(
                        test["Sales"],
                        test_predictions
                    )

                    st.subheader("Model Evaluation")

                    col1, col2, col3 = st.columns(3)

                    col1.metric("MAE", round(mae, 2))
                    col2.metric("RMSE", round(rmse, 2))
                    col3.metric("R² Score", round(r2, 2))

                    st.subheader("Actual vs Predicted Sales")

                    comparison = pd.DataFrame({
                        "Actual Sales": test["Sales"].values,
                        "Predicted Sales": test_predictions
                    })

                    st.line_chart(comparison)

                    # FUTURE DEMAND FORECAST

                    st.subheader("Future Demand Forecast")

                    forecast_months = st.slider(
                        "Select forecast months",
                        1, 12, 6
                    )

                    model = LinearRegression()

                    model.fit(
                        monthly[["Time"]],
                        monthly["Sales"]
                    )

                    future_time = pd.DataFrame({
                        "Time": range(
                            len(monthly) + 1,
                            len(monthly) + forecast_months + 1
                        )
                    })

                    predictions = model.predict(future_time)

                    future_dates = pd.date_range(
                        start=monthly["Date"].max(),
                        periods=forecast_months + 1,
                        freq="ME"
                    )[1:]

                    forecast = pd.DataFrame({
                        "Date": future_dates,
                        "Predicted Sales": predictions.clip(min=0)
                    })

                    st.dataframe(forecast)

                    st.line_chart(
                        forecast.set_index("Date")
                    )

                    st.download_button(
                        "Download Forecast CSV",
                        forecast.to_csv(index=False),
                        "Sales_Forecast.csv",
                        "text/csv"
                    )

    except Exception as e:
        st.error(f"Error: {e}")
                    
