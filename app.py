
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
    page_title="Sales Demand Forecasting",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Sales Demand Forecasting System")
st.write("Analyze historical sales and predict future demand.")

# Upload CSV
uploaded_file = st.file_uploader(
    "Upload Sales CSV File",
    type=["csv"]
)

if uploaded_file is not None:

    data = pd.read_csv(uploaded_file)

    # Automatic column name detection
    if 'Order Date' in data.columns:
        data = data.rename(
            columns={'Order Date': 'Date'}
        )

    # Check required columns
    if not {'Date', 'Sales'}.issubset(data.columns):

        st.error(
            "CSV must contain Date (or Order Date) and Sales columns."
        )

    else:

        # Data Cleaning
        data['Date'] = pd.to_datetime(
            data['Date'], errors='coerce'
        )

        data['Sales'] = pd.to_numeric(
            data['Sales'], errors='coerce'
        )

        data = data.dropna(
            subset=['Date', 'Sales']
        )

        data = data.sort_values('Date')

        # Monthly Sales Data
        monthly = (
            data.set_index('Date')['Sales']
            .resample('MS')
            .sum()
            .reset_index()
        )

        if len(monthly) < 12:

            st.error(
                "Please upload at least 12 months of valid sales data."
            )

        else:

            monthly['Time'] = range(
                1, len(monthly) + 1
            )

            # Sales Overview
            st.subheader("Sales Overview")

            col1, col2 = st.columns(2)

            col1.metric(
                "Total Sales",
                f"{monthly['Sales'].sum():,.2f}"
            )

            col2.metric(
                "Total Months",
                len(monthly)
            )

            # Monthly Sales Trend
            st.subheader("Monthly Sales Trend")

            st.line_chart(
                monthly.set_index('Date')['Sales']
            )

            # Train Test Split
            train = monthly.iloc[:-6]
            test = monthly.iloc[-6:]

            model = LinearRegression()

            model.fit(
                train[['Time']],
                train['Sales']
            )

            predictions = model.predict(
                test[['Time']]
            )

            # Model Evaluation
            mae = mean_absolute_error(
                test['Sales'],
                predictions
            )

            rmse = mean_squared_error(
                test['Sales'],
                predictions
            ) ** 0.5

            r2 = r2_score(
                test['Sales'],
                predictions
            )

            st.subheader("Model Evaluation")

            col1, col2, col3 = st.columns(3)

            col1.metric("MAE", f"{mae:,.2f}")
            col2.metric("RMSE", f"{rmse:,.2f}")
            col3.metric("R² Score", f"{r2:.2f}")

            # Actual vs Predicted Sales
            st.subheader("Actual vs Predicted Sales")

            comparison = pd.DataFrame({
                'Actual Sales': test['Sales'].values,
                'Predicted Sales': predictions
            }, index=test['Date'])

            st.line_chart(comparison)

            # Final Model
            final_model = LinearRegression()

            final_model.fit(
                monthly[['Time']],
                monthly['Sales']
            )

            # Future Forecast
            st.subheader("Future Demand Forecast")

            periods = st.slider(
                "Select Forecast Months",
                1, 12, 6
            )

            future_time = pd.DataFrame({
                'Time': range(
                    len(monthly) + 1,
                    len(monthly) + periods + 1
                )
            })

            future_dates = pd.date_range(
                monthly['Date'].max()
                + pd.offsets.MonthBegin(1),
                periods=periods,
                freq='MS'
            )

            forecast = pd.DataFrame({
                'Date': future_dates,
                'Predicted Sales': final_model.predict(
                    future_time
                ).round(2)
            })

            st.dataframe(
                forecast,
                use_container_width=True
            )

            # Forecast Chart
            st.line_chart(
                forecast.set_index('Date')
            )

            # Download Forecast
            st.download_button(
                "📥 Download Forecast CSV",
                forecast.to_csv(index=False),
                "Sales_Forecast.csv",
                "text/csv"
            )

else:

    st.info("Please upload your Sales CSV file.")
        
