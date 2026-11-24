import streamlit as st
import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# PAGE CONFIGURATION

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

# HELPER FUNCTIONS


def create_features(dates, start_time=1):

    dates = pd.DatetimeIndex(
        pd.to_datetime(dates)
    )

    return pd.DataFrame({
        "Time": np.arange(
            start_time,
            start_time + len(dates)
        ),
        "MonthSin": np.sin(
            2 * np.pi * dates.month / 12
        ),
        "MonthCos": np.cos(
            2 * np.pi * dates.month / 12
        )
    })
    

def calculate_metrics(actual, predicted):

    mae = mean_absolute_error(actual, predicted)

    rmse = np.sqrt(
        mean_squared_error(actual, predicted)
    )

    r2 = r2_score(actual, predicted)

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


def make_monthly_data(frame):

    monthly_series = (
        frame
        .set_index("Date")["Value"]
        .resample("MS")
        .sum()
    )

    full_months = pd.date_range(
        monthly_series.index.min(),
        monthly_series.index.max(),
        freq="MS"
    )

    monthly_series = monthly_series.reindex(
        full_months,
        fill_value=0
    )

    monthly = monthly_series.rename("Value").reset_index()

    monthly.columns = ["Date", "Value"]

    return monthly


# DATASET UPLOAD

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

    if len(data.columns) == 0:
        st.error("No columns found in the uploaded CSV.")
        st.stop()

    # DATASET OVERVIEW

    st.header("1. Dataset Overview")

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Records", len(data))

    col2.metric("Total Columns", len(data.columns))

    col3.metric(
        "Missing Values",
        int(data.isnull().sum().sum())
    )

    st.subheader("Dataset Preview")

    st.dataframe(
        data.head(10),
        use_container_width=True
    )
    
    # COLUMN DETECTION

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
        if (
            str(col).strip().lower() in date_names
            or "date" in str(col).strip().lower()
        )
    ]

    numeric_columns = []

    for col in data.columns:

        converted = pd.to_numeric(
            data[col],
            errors="coerce"
        )

        if converted.notna().sum() > 0:
            numeric_columns.append(col)

    # DATE COLUMN

    date_options = ["Not selected"] + list(data.columns)

    date_default = (
        date_options.index(date_candidates[0])
        if date_candidates
        else 0
    )

    date_col = st.selectbox(
        "Select Date Column",
        date_options,
        index=date_default,
        key="main_date"
    )

    # NUMERIC COLUMN

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

    # CATEGORY COLUMN

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

    # BUSINESS ANALYTICS

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

            st.subheader(f"{metric_col} Distribution")

            st.bar_chart(values.head(20))

            # CATEGORY ANALYSIS

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
    
    
    # OVERALL DEMAND FORECASTING

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

            monthly = make_monthly_data(forecast_data)

            st.subheader("Monthly Trend")

            st.line_chart(
                monthly.set_index("Date")["Value"]
            )

            if len(monthly) >= 12:

                # PREPARE FEATURES

                features = create_features(
                    monthly["Date"]
                )

                monthly["Time"] = features["Time"]
                monthly["MonthSin"] = features["MonthSin"]
                monthly["MonthCos"] = features["MonthCos"]

                feature_cols = [
                    "Time",
                    "MonthSin",
                    "MonthCos"
                ]

                # TRAIN AND TEST DATA

                train = monthly.iloc[:-6].copy()
                test = monthly.iloc[-6:].copy()

                X_train = train[feature_cols]
                y_train = train["Value"]

                X_test = test[feature_cols]
                y_test = test["Value"]

                # MODEL COMPARISON

                st.subheader("Model Comparison")

                models = get_models()

                evaluation_results = []
                predictions_dict = {}

                for model_name, model in models.items():

                    model.fit(X_train, y_train)

                    predictions = model.predict(X_test)

                    mae, rmse, r2 = calculate_metrics(
                        y_test,
                        predictions
                    )

                    evaluation_results.append({
                        "Model": model_name,
                        "MAE": round(mae, 2),
                        "RMSE": round(rmse, 2),
                        "R2 Score": round(r2, 3)
                    })

                    predictions_dict[model_name] = predictions

                # SEASONAL NAIVE

                if len(monthly) >= 18:

                    seasonal_predictions = (
                        monthly["Value"].iloc[-12:-6].values
                    )

                    mae, rmse, r2 = calculate_metrics(
                        y_test,
                        seasonal_predictions
                    )

                    evaluation_results.append({
                        "Model": "Seasonal Naive",
                        "MAE": round(mae, 2),
                        "RMSE": round(rmse, 2),
                        "R2 Score": round(r2, 3)
                    })

                evaluation_df = pd.DataFrame(
                    evaluation_results
                )

                # SELECT BEST MODEL

                best_index = evaluation_df["MAE"].idxmin()

                best_model_name = evaluation_df.loc[
                    best_index,
                    "Model"
                ]

                st.dataframe(
                    evaluation_df.sort_values("MAE"),
                    use_container_width=True
                )

                st.success(
                    f"Best Model Selected: {best_model_name}"
                )

                best_row = evaluation_df.loc[best_index]

                col1, col2, col3 = st.columns(3)

                col1.metric(
                    "Best Model MAE",
                    f"{best_row['MAE']:,.2f}"
                )

                col2.metric(
                    "Best Model RMSE",
                    f"{best_row['RMSE']:,.2f}"
                )

                col3.metric(
                    "Best Model R2",
                    f"{best_row['R2 Score']:.3f}"
                )

                # ACTUAL VS PREDICTED

                st.subheader("Actual vs Predicted")

                if best_model_name == "Seasonal Naive":

                    best_predictions = seasonal_predictions

                else:

                    best_predictions = predictions_dict[
                        best_model_name
                    ]

                comparison = pd.DataFrame({
                    "Actual": y_test.values,
                    "Predicted": best_predictions
                }, index=test["Date"])

                st.line_chart(comparison)

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
        
    # FUTURE DEMAND FORECAST

    st.subheader("Future Demand Forecast")

    if (
        date_col != "Not selected"
        and metric_col != "Not selected"
        and not forecast_data.empty
        and len(monthly) >= 12
    ):

        forecast_months = st.slider(
            "Number of Future Months",
            min_value=1,
            max_value=12,
            value=6,
            key="overall_forecast_months"
        )

        future_dates = pd.date_range(
            start=monthly["Date"].max() + pd.offsets.MonthBegin(1),
            periods=forecast_months,
            freq="MS"
        )

        future_features = create_features(
            future_dates,
            start_time=len(monthly) + 1
        )

        future_predictions = []

        if best_model_name == "Seasonal Naive":

            for i in range(forecast_months):

                value = monthly["Value"].iloc[
                    -12 + (i % 12)
                ]

                future_predictions.append(value)

        else:

            final_model = get_models()[best_model_name]

            final_model.fit(
                monthly[feature_cols],
                monthly["Value"]
            )

            future_predictions = final_model.predict(
                future_features[feature_cols]
            )

        future_df = pd.DataFrame({
            "Date": future_dates,
            "Predicted Demand": np.maximum(
                future_predictions,
                0
            )
        })

        st.dataframe(
            future_df.round(2),
            use_container_width=True
        )

        st.line_chart(
            future_df.set_index("Date")
        )

        st.download_button(
            label="Download Overall Forecast CSV",
            data=future_df.to_csv(index=False),
            file_name="Overall_Demand_Forecast.csv",
            mime="text/csv"
        )

    # PRODUCT-WISE FORECASTING SETUP

    st.header("5. Product-wise Demand Forecasting")

    st.write(
        "Select the product name and demand columns "
        "to forecast demand for individual products."
    )

    product_columns = [
        col for col in data.columns
        if col != date_col
        and col not in numeric_columns
    ]

    product_options = [
        "Not selected"
    ] + product_columns

    product_default = 0

    for i, col in enumerate(product_options):

        if str(col).strip().lower() in [
            "product",
            "product name",
            "item",
            "item name"
        ]:
            product_default = i
            break

    product_col = st.selectbox(
        "Select Product Name Column",
        product_options,
        index=product_default,
        key="product_name_column"
    )

    demand_default = 0

    for i, col in enumerate(numeric_columns):

        if str(col).strip().lower() in [
            "quantity",
            "demand",
            "units sold"
        ]:
            demand_default = i
            break

    demand_col = "Not selected"

    if numeric_columns:

        demand_col = st.selectbox(
            "Select Product Demand Column",
            numeric_columns,
            index=demand_default,
            key="product_demand_column"
        )

    else:

        st.warning(
            "No numeric demand column found."
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
                "Product": data[product_col].astype(str),
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

                first_month = (
                    product_data["Date"]
                    .min()
                    .to_period("M")
                    .to_timestamp()
                )

                last_month = (
                    product_data["Date"]
                    .max()
                    .to_period("M")
                    .to_timestamp()
                )

                all_months = pd.date_range(
                    start=first_month,
                    end=last_month,
                    freq="MS"
                )

                product_results = []
                future_results = []

                progress = st.progress(0)

                products = sorted(
                    product_data["Product"].unique()
                )

                for product_number, product_name in enumerate(products):

                    product_frame = product_data[
                        product_data["Product"] == product_name
                    ].copy()

                    monthly_series = (
                        product_frame
                        .set_index("Date")["Value"]
                        .resample("MS")
                        .sum()
                        .reindex(all_months, fill_value=0)
                    )

                    if len(monthly_series) < 12:
                        continue

                    monthly_product = pd.DataFrame({
                        "Date": all_months,
                        "Value": monthly_series.values
                    })

                    product_features = create_features(
                        monthly_product["Date"]
                    )

                    feature_columns = [
                        "Time",
                        "MonthSin",
                        "MonthCos"
                    ]

                    train = monthly_product.iloc[:-6]
                    test = monthly_product.iloc[-6:]

                    X_train = product_features.iloc[:-6][
                        feature_columns
                    ]

                    X_test = product_features.iloc[-6:][
                        feature_columns
                    ]

                    y_train = train["Value"]
                    y_test = test["Value"]

                    model_scores = []
                    prediction_store = {}

                    # LINEAR REGRESSION AND RANDOM FOREST

                    for model_name, model in get_models().items():

                        model.fit(X_train, y_train)

                        predictions = model.predict(X_test)

                        mae, rmse, r2 = calculate_metrics(
                            y_test,
                            predictions
                        )

                        model_scores.append({
                            "Model": model_name,
                            "MAE": mae,
                            "RMSE": rmse,
                            "R2 Score": r2
                        })

                        prediction_store[model_name] = predictions

                    # SEASONAL NAIVE MODEL

                    if len(monthly_product) >= 18:

                        seasonal_predictions = (
                            monthly_product["Value"]
                            .iloc[-12:-6]
                            .values
                        )

                        mae, rmse, r2 = calculate_metrics(
                            y_test,
                            seasonal_predictions
                        )

                        model_scores.append({
                            "Model": "Seasonal Naive",
                            "MAE": mae,
                            "RMSE": rmse,
                            "R2 Score": r2
                        })

                        prediction_store[
                            "Seasonal Naive"
                        ] = seasonal_predictions

                    scores_df = pd.DataFrame(model_scores)

                    best_index = scores_df["MAE"].idxmin()

                    best_model_name = scores_df.loc[
                        best_index,
                        "Model"
                    ]

                    best_scores = scores_df.loc[best_index]

                    product_results.append({
                        "Product": product_name,
                        "Best Model": best_model_name,
                        "MAE": round(best_scores["MAE"], 2),
                        "RMSE": round(best_scores["RMSE"], 2),
                        "R2 Score": round(best_scores["R2 Score"], 3)
                    })

                    # FUTURE PREDICTION

                    future_dates = pd.date_range(
                        start=all_months[-1] + pd.offsets.MonthBegin(1),
                        periods=forecast_period,
                        freq="MS"
                    )

                    if best_model_name == "Seasonal Naive":

                        future_predictions = [
                            monthly_product["Value"].iloc[
                                -12 + (i % 12)
                            ]
                            for i in range(forecast_period)
                        ]

                    else:

                        final_model = get_models()[best_model_name]

                        final_model.fit(
                            product_features[feature_columns],
                            monthly_product["Value"]
                        )

                        future_features = create_features(
                            future_dates,
                            start_time=len(monthly_product) + 1
                        )

                        future_predictions = final_model.predict(
                            future_features[feature_columns]
                        )

                    for forecast_date, prediction in zip(
                        future_dates,
                        future_predictions
                    ):

                        future_results.append({
                            "Product": product_name,
                            "Date": forecast_date,
                            "Predicted Demand": round(
                                max(0, float(prediction)),
                                2
                            )
                        })

                    progress.progress(
                        (product_number + 1) / len(products)
                    )

                progress.empty()

                # DISPLAY RESULTS

                if product_results:

                    result_df = pd.DataFrame(product_results)

                    future_df = pd.DataFrame(future_results)

                    st.success(
                        f"Forecast completed for "
                        f"{len(result_df)} products."
                    )

                    st.subheader("Product-wise Model Evaluation")

                    st.dataframe(
                        result_df.sort_values("MAE"),
                        use_container_width=True
                    )

                    st.subheader("Future Product-wise Demand Forecast")

                    st.dataframe(
                        future_df,
                        use_container_width=True
                    )

                    st.subheader("Product-wise Forecast Chart")

                    chart_data = future_df.pivot(
                        index="Date",
                        columns="Product",
                        values="Predicted Demand"
                    )

                    st.line_chart(chart_data)

                    # PRODUCT RANKING

                    st.subheader("Product Demand Ranking")

                    ranking = (
                        future_df
                        .groupby("Product")["Predicted Demand"]
                        .sum()
                        .sort_values(ascending=False)
                        .reset_index()
                    )

                    ranking.columns = [
                        "Product",
                        "Total Predicted Demand"
                    ]

                    ranking.insert(
                        0,
                        "Rank",
                        range(1, len(ranking) + 1)
                    )

                    st.dataframe(
                        ranking,
                        use_container_width=True
                    )

                    st.success(
                        "Top Forecasted Product: "
                        + str(ranking.iloc[0]["Product"])
                    )

                    # DOWNLOAD FILES

                    st.download_button(
                        label="Download Product-wise Forecast CSV",
                        data=future_df.to_csv(index=False),
                        file_name="Product_Wise_Forecast.csv",
                        mime="text/csv"
                    )

                    st.download_button(
                        label="Download Product Model Evaluation CSV",
                        data=result_df.to_csv(index=False),
                        file_name="Product_Model_Evaluation.csv",
                        mime="text/csv"
                    )

                else:

                    st.warning(
                        "No products have at least 12 monthly "
                        "periods of data. Try a dataset with "
                        "more historical records."
                    )

    else:

        st.info(
            "Select Date, Product Name and Demand columns "
            "to enable product-wise forecasting."
    )
                    
