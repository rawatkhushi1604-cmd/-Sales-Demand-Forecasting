
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

st.set_page_config(
    page_title="Sales Demand Forecasting",
    page_icon="📊",
    layout="wide"
)

st.title("Sales Demand Forecasting System")
st.write("Analyze sales and predict future demand.")

uploaded_file = st.file_uploader(
    "Upload Sales CSV File",
    type=["csv"]
)

if uploaded_file is not None:

    data = pd.read_csv(uploaded_file)

    if not {'Date', 'Sales'}.issubset(data.columns):
        st.error("CSV must contain Date and Sales columns.")

    else:
        data['Date'] = pd.to_datetime(
            data['Date'], errors='coerce'
        )
        data['Sales'] = pd.to_numeric(
            data['Sales'], errors='coerce'
        )

        data = data.dropna(subset=['Date', 'Sales'])

        if len(data) < 12:
            st.error("Please upload at least 12 valid records.")

        else:
            monthly = (
                data.set_index('Date')['Sales']
                .resample('MS').sum()
                .reset_index()
            )

            monthly['Time'] = range(1, len(monthly) + 1)

            st.subheader("Sales Overview")

            col1, col2 = st.columns(2)

            col1.metric(
                "Total Sales",
                round(monthly['Sales'].sum(), 2)
            )

            col2.metric(
                "Total Months",
                len(monthly)
            )

            st.subheader("Monthly Sales Trend")

            st.line_chart(
                monthly.set_index('Date')['Sales']
            )

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

            mae = mean_absolute_error(
                test['Sales'],
                predictions
            )

            st.subheader("Model Evaluation")

            st.metric("Mean Absolute Error", round(mae, 2))

            final_model = LinearRegression()

            final_model.fit(
                monthly[['Time']],
                monthly['Sales']
            )

            periods = st.slider(
                "Forecast Months",
                1, 12, 6
            )

            future_time = pd.DataFrame({
                'Time': range(
                    len(monthly) + 1,
                    len(monthly) + periods + 1
                )
            })

            future_dates = pd.date_range(
                monthly['Date'].max() + pd.offsets.MonthBegin(1),
                periods=periods,
                freq='MS'
            )

            forecast = pd.DataFrame({
                'Date': future_dates,
                'Forecasted Sales': final_model.predict(
                    future_time
                ).round(2)
            })

            st.subheader("Future Demand Forecast")

            st.dataframe(forecast)

            st.line_chart(
                forecast.set_index('Date')
            )

            st.download_button(
                "Download Forecast CSV",
                forecast.to_csv(index=False),
                "Sales_Forecast.csv",
                "text/csv"
            )

else:
    st.info("Please upload your Sales CSV file.")
