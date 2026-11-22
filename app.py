
import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.arima.model import ARIMA

# ==========================================================
# 1. PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="AgriVanta AI",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================================
# 2. PROJECT PATHS
# ==========================================================

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

BASE_DIR = os.path.join(PROJECT_DIR, "lankaprice_models")

MODEL_PATH = os.path.join(BASE_DIR, "final_random_forest.pkl")
HISTORY_PATH = os.path.join(BASE_DIR, "market_history.csv")
METRICS_PATH = os.path.join(BASE_DIR, "model_comparison.csv")

DATASET_PATH = os.path.join(
    PROJECT_DIR,
    "lankaprice_data",
    "LKA_RTFP_mkt_2007_2026-11-17.csv"
)

# ==========================================================
# 3. PROFESSIONAL LIGHT UI DESIGN
# ==========================================================

st.markdown("""
<style>

/* MAIN BACKGROUND */

.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMain"] {
    background-color: #f4f7fb !important;
    color: #243b53 !important;
    color-scheme: light !important;
}

/* MAIN TEXT */

[data-testid="stMain"] p,
[data-testid="stMain"] li,
[data-testid="stMain"] label,
[data-testid="stMain"] span,
[data-testid="stMain"] small {
    color: #243b53;
}

/* HEADINGS */

[data-testid="stMain"] h1,
[data-testid="stMain"] h2,
[data-testid="stMain"] h3,
[data-testid="stMain"] h4 {
    color: #102a43 !important;
}

/* HEADER */

[data-testid="stHeader"] {
    background-color: #102a43 !important;
}

[data-testid="stHeader"] button {
    color: #ffffff !important;
}

/* SIDEBAR */

[data-testid="stSidebar"],
[data-testid="stSidebar"] > div:first-child {
    background-color: #102a43 !important;
}

[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] small,
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #ffffff !important;
}

/* SIDEBAR BRAND */

.sidebar-brand {
    font-size: 27px;
    font-weight: 800;
    color: #ffffff !important;
    padding: 12px 5px 0 5px;
}

.sidebar-description {
    font-size: 12px;
    color: #b9c9d8 !important;
    padding-left: 5px;
    margin-bottom: 20px;
}

/* SIDEBAR NAVIGATION BUTTONS */

[data-testid="stSidebar"] button {
    width: 100%;
    background-color: #183b56 !important;
    color: #ffffff !important;
    border: 1px solid #31536d !important;
    border-radius: 10px !important;
    padding: 12px 14px !important;
    margin: 4px 0;
    transition: all 0.2s ease-in-out;
}

[data-testid="stSidebar"] button:hover {
    background-color: #287271 !important;
    border: 1px solid #5eead4 !important;
    transform: translateX(4px);
}

[data-testid="stSidebar"] button[kind="primary"] {
    background-color: #287271 !important;
    border: 1px solid #5eead4 !important;
    color: #ffffff !important;
    font-weight: 700 !important;
}

/* INPUTS */

[data-testid="stMain"] input,
[data-testid="stMain"] textarea {
    background-color: #ffffff !important;
    color: #102a43 !important;
    border-color: #cbd5e1 !important;
}

[data-testid="stMain"] [data-baseweb="select"] > div {
    background-color: #ffffff !important;
    color: #102a43 !important;
    border-color: #cbd5e1 !important;
}

[data-testid="stMain"] [data-baseweb="select"] * {
    color: #102a43 !important;
}

/* DROPDOWN */

[data-baseweb="popover"],
[data-baseweb="menu"] {
    background-color: #ffffff !important;
}

[data-baseweb="menu"] * {
    color: #102a43 !important;
}

/* METRIC CARDS */

[data-testid="stMetric"] {
    background-color: #ffffff !important;
    border: 1px solid #dce5ef !important;
    border-radius: 13px !important;
    padding: 18px !important;
    box-shadow: 0 2px 7px rgba(16,42,67,0.04);
}

[data-testid="stMetricLabel"] *,
[data-testid="stMetricValue"] *,
[data-testid="stMetricDelta"] * {
    color: #102a43 !important;
}

/* DATA TABLES */

[data-testid="stDataFrame"],
[data-testid="stTable"] {
    background-color: #ffffff !important;
    color: #102a43 !important;
}

/* GENERAL MAIN BUTTONS */

[data-testid="stMain"] .stButton button {
    background-color: #176b63 !important;
    color: #ffffff !important;
    border: 1px solid #176b63 !important;
    border-radius: 9px !important;
    font-weight: 600 !important;
}

/* MAIN BUTTON HOVER */

[data-testid="stMain"] .stButton button:hover {
    background-color: #12534d !important;
    color: #ffffff !important;
}

/* GENERATE PREDICTION PRIMARY BUTTON */

[data-testid="stMain"] .stButton button[kind="primary"] {
    background-color: #168653 !important;
    border: 1px solid #168653 !important;
    color: #ffffff !important;
    font-weight: 700 !important;
}

[data-testid="stMain"] .stButton button[kind="primary"]:hover {
    background-color: #106B41 !important;
    border: 1px solid #106B41 !important;
    color: #ffffff !important;
}

/* DOWNLOAD BUTTONS */

[data-testid="stDownloadButton"] button {
    background-color: #168653 !important;
    color: #ffffff !important;
    border: 1px solid #168653 !important;
    border-radius: 9px !important;
    font-weight: 600 !important;
}

[data-testid="stDownloadButton"] button:hover {
    background-color: #106B41 !important;
    color: #ffffff !important;
}

/* ALERTS */

[data-testid="stAlert"] {
    border-radius: 10px !important;
}

/* PAGE TITLES */

.main-title {
    font-size: 36px;
    font-weight: 800;
    color: #102a43 !important;
    margin-bottom: 3px;
}

.subtitle {
    color: #526b82 !important;
    font-size: 15px;
    margin-bottom: 25px;
}

/* SECTION HEADINGS */

.section-heading {
    font-size: 23px;
    font-weight: 750;
    color: #102a43 !important;
    margin-top: 15px;
    margin-bottom: 12px;
}

/* FOOTER */

.footer {
    text-align: center;
    color: #526b82 !important;
    font-size: 12px;
    padding: 20px;
}

</style>
""", unsafe_allow_html=True)


# ==========================================================
# 4. LOAD PROJECT FILES
# ==========================================================

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_history():
    data = pd.read_csv(HISTORY_PATH)
    data["price_date"] = pd.to_datetime(data["price_date"])
    return data


@st.cache_data
def load_metrics():
    return pd.read_csv(METRICS_PATH)


try:
    model = load_model()
    history = load_history()
    metrics = load_metrics()

except Exception as e:
    st.error(f"Unable to load project files: {e}")
    st.info(
        "Check that the lankaprice_models folder contains "
        "all three saved files."
    )
    st.stop()


# ==========================================================
# 5. LOAD ORIGINAL DATASET FOR DOWNLOAD
# ==========================================================

def original_dataset_download():

    if os.path.exists(DATASET_PATH):

        with open(DATASET_PATH, "rb") as file:
            return file.read()

    return None


original_dataset = original_dataset_download()


# ==========================================================
# 6. COMMON PROJECT INFORMATION
# ==========================================================

required_columns = [
    "price_date",
    "mkt_name",
    "c_rice_various"
]

if not all(col in history.columns for col in required_columns):
    st.error("The historical data file is missing required columns.")
    st.stop()

if history.empty:
    st.error("The historical dataset is empty.")
    st.stop()

latest_date = history["price_date"].max()

market_list = sorted(history["mkt_name"].dropna().unique())

number_markets = len(market_list)
number_records = len(history)
average_price = history["c_rice_various"].mean()


# ==========================================================
# 7. FIND BEST MODEL AUTOMATICALLY
# ==========================================================

best_model_row = None

if {"Model", "MAE"}.issubset(metrics.columns):

    valid_metrics = metrics.dropna(subset=["MAE"])

    if not valid_metrics.empty:
        best_model_row = valid_metrics.loc[
            valid_metrics["MAE"].idxmin()
        ]


# ==========================================================
# 8. SIDEBAR NAVIGATION
# ==========================================================

st.sidebar.markdown(
    '<div class="sidebar-brand">AgriVanta AI</div>',
    unsafe_allow_html=True
)

st.sidebar.markdown(
    '<div class="sidebar-description">'
    'AGRICULTURAL PRICE INTELLIGENCE'
    '</div>',
    unsafe_allow_html=True
)

st.sidebar.markdown("---")

pages = {
    "Overview": "Overview",
    "Price Prediction": "Price Prediction",
    "Historical Analysis": "Historical Analysis",
    "Model Performance": "Model Performance",
    "Feature Importance": "Feature Importance",
    "Data Explorer": "Data Explorer"
}

if "selected_page" not in st.session_state:
    st.session_state.selected_page = "Overview"

for label, page_name in pages.items():

    button_type = (
        "primary"
        if st.session_state.selected_page == page_name
        else "secondary"
    )

    if st.sidebar.button(
        label,
        key=f"nav_{page_name}",
        width="stretch",
        type=button_type
    ):
        st.session_state.selected_page = page_name
        st.rerun()

page = st.session_state.selected_page

st.sidebar.markdown("---")
st.sidebar.markdown("### Project Details")
st.sidebar.caption("Country: Sri Lanka")
st.sidebar.caption("Commodity: Various Rice")
st.sidebar.caption("Currency: LKR")
st.sidebar.caption("Source: World Bank")
st.sidebar.markdown("---")
st.sidebar.caption("Machine Learning Academic Project")


# ==========================================================
# 9. COMMON FUNCTIONS
# ==========================================================

def page_header(title, description):

    st.markdown(
        f'<div class="main-title">{title}</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        f'<div class="subtitle">{description}</div>',
        unsafe_allow_html=True
    )


def section(title):

    st.markdown(
        f'<div class="section-heading">{title}</div>',
        unsafe_allow_html=True
    )


def show_original_dataset_download():

    if original_dataset is not None:

        st.download_button(
            label="Download Original World Bank Dataset",
            data=original_dataset,
            file_name="LKA_RTFP_mkt_2007_2026-11-17.csv",
            mime="text/csv",
            width="stretch"
        )

    else:

        st.warning(
            "Original dataset file not found. "
            "Please check the lankaprice_data folder."
        )


# ==========================================================
# 10. PRICE PREDICTION FUNCTION
# ==========================================================

def make_forecast(selected_market):

    market_data = (
        history[history["mkt_name"] == selected_market]
        .sort_values("price_date")
        .copy()
    )

    values = (
        market_data["c_rice_various"]
        .astype(float)
        .values
    )

    if len(values) < 13:
        raise ValueError(
            "Not enough historical records for this market."
        )

    market_latest_date = market_data["price_date"].max()

    next_date = market_latest_date + pd.DateOffset(months=1)

    feature_row = pd.DataFrame([{
        "mkt_name": selected_market,
        "year": next_date.year,
        "month": next_date.month,
        "lag_1": values[-1],
        "lag_2": values[-2],
        "lag_3": values[-3],
        "lag_6": values[-6],
        "lag_12": values[-12],
        "rolling_mean_3": np.mean(values[-3:]),
        "rolling_mean_6": np.mean(values[-6:])
    }])

    rf_prediction = float(model.predict(feature_row)[0])

    arima_prediction = None
    hybrid_prediction = None

    try:

        arima_fit = ARIMA(
            values,
            order=(1, 1, 1)
        ).fit()

        arima_prediction = float(
            arima_fit.forecast(steps=1)[0]
        )

        hybrid_prediction = (
            rf_prediction + arima_prediction
        ) / 2

    except Exception:
        pass

    return {
        "date": next_date,
        "rf": rf_prediction,
        "arima": arima_prediction,
        "hybrid": hybrid_prediction,
        "latest": float(values[-1]),
        "market_data": market_data
    }


# ==========================================================
# 11. PAGE: OVERVIEW
# ==========================================================

if page == "Overview":

    page_header(
        "AgriVanta AI",
        "Intelligent Food Commodity Price Prediction "
        "and Market Analysis in Sri Lanka"
    )

    st.info(
        "An intelligent price analysis platform using "
        "machine learning and time-series forecasting "
        "to study Sri Lankan rice price estimates."
    )

    # GENERAL PROJECT METRICS

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Markets", f"{number_markets}")
    c2.metric("Historical Records", f"{number_records:,}")
    c3.metric("Average Estimate", f"Rs. {average_price:,.2f}")
    c4.metric("Latest Data", latest_date.strftime("%b %Y"))

    # BEST MODEL SECTION

    section("Best Performing Model")

    if best_model_row is not None:

        best_name = str(best_model_row["Model"])
        best_mae = float(best_model_row["MAE"])

        st.success(
            f"Best test model: {best_name}"
        )

        best_columns = st.columns(3)

        best_columns[0].metric(
            "Best Model",
            best_name
        )

        best_columns[1].metric(
            "MAE",
            f"{best_mae:.4f}"
        )

        if "RMSE" in metrics.columns:

            best_columns[2].metric(
                "RMSE",
                f"{float(best_model_row['RMSE']):.4f}"
            )

        if "R2" in metrics.columns:

            st.metric(
                "R² Score",
                f"{float(best_model_row['R2']):.4f}"
            )

    else:

        st.warning(
            "Best model information is unavailable. "
            "Check the columns in model_comparison.csv."
        )

    st.caption(
        "The best model is identified using the lowest MAE "
        "in the model evaluation results. This describes "
        "test performance, not necessarily the model used "
        "for every dashboard forecast."
    )

    # PROJECT OVERVIEW

    section("Project Overview")

    left, right = st.columns(2)

    with left:

        st.markdown("""
        ### What is AgriVanta AI?

        AgriVanta AI analyses monthly rice price estimates
        across Sri Lankan markets.

        **Main Objectives**

        - Analyse historical rice price patterns.
        - Predict next month's estimated rice price.
        - Compare machine learning models.
        - Explore differences between markets.
        """)

    with right:

        st.markdown("""
        ### Machine Learning Models

        - Random Forest
        - XGBoost
        - LightGBM
        - CatBoost
        - Stacking
        - ARIMA
        - Hybrid Random Forest + ARIMA
        """)

    # HISTORICAL OVERVIEW

    section("Historical Price Overview")

    monthly = (
        history.groupby("price_date")["c_rice_various"]
        .mean()
        .reset_index()
    )

    fig = px.line(
        monthly,
        x="price_date",
        y="c_rice_various",
        title="Average Monthly Rice Price Estimate",
        labels={
            "price_date": "Date",
            "c_rice_various": "Estimated Price (LKR)"
        }
    )

    fig.update_layout(
        template="plotly_white",
        height=430,
        font=dict(color="#243b53"),
        title_font_color="#102a43"
    )

    st.plotly_chart(fig, width="stretch")

    st.caption(
        "Source: World Bank Microdata Library. "
        "Values are modeled monthly price estimates, "
        "not necessarily direct market observations."
    )

    # ORIGINAL DATASET DOWNLOAD

    section("Download Original Dataset")

    st.write(
        "Download the original World Bank Sri Lankan "
        "food price dataset used in this project."
    )

    show_original_dataset_download()


# ==========================================================
# 12. PAGE: PRICE PREDICTION
# ==========================================================

elif page == "Price Prediction":

    page_header(
        "Rice Price Prediction",
        "Generate next-month rice price estimates "
        "for a selected Sri Lankan market."
    )

    selected_market = st.selectbox(
        "Select Sri Lankan Market",
        market_list
    )

    if st.button(
        "Generate Price Prediction",
        type="primary",
        width="stretch"
    ):

        try:

            result = make_forecast(selected_market)

            st.success(
                f"Prediction generated for "
                f"{result['date'].strftime('%B %Y')}"
            )

            section("Prediction Results")

            st.write(f"**Selected Market:** {selected_market}")

            c1, c2, c3 = st.columns(3)

            c1.metric(
                "Random Forest",
                f"Rs. {result['rf']:,.2f}"
            )

            if result["arima"] is not None:

                c2.metric(
                    "ARIMA",
                    f"Rs. {result['arima']:,.2f}"
                )

                c3.metric(
                    "Hybrid Prediction",
                    f"Rs. {result['hybrid']:,.2f}"
                )

            else:

                c2.metric("ARIMA", "Unavailable")
                c3.metric("Hybrid", "Unavailable")

            section("Prediction Explanation")

            st.write(
                f"The latest available modeled estimate is "
                f"Rs. {result['latest']:,.2f}. "
                f"The Random Forest model estimates the next "
                f"month's value at Rs. {result['rf']:,.2f}."
            )

            if result["hybrid"] is not None:

                st.write(
                    f"The hybrid model combines Random Forest "
                    f"and ARIMA equally, producing an estimate "
                    f"of Rs. {result['hybrid']:,.2f}."
                )

            recent = result["market_data"].tail(36)

            fig = px.line(
                recent,
                x="price_date",
                y="c_rice_various",
                title=f"Recent Price History - {selected_market}",
                labels={
                    "price_date": "Date",
                    "c_rice_various": "Estimated Price (LKR)"
                }
            )

            fig.add_trace(
                go.Scatter(
                    x=[result["date"]],
                    y=[result["rf"]],
                    mode="markers",
                    name="RF Forecast",
                    marker=dict(size=13, symbol="star")
                )
            )

            fig.update_layout(
                template="plotly_white",
                height=430,
                font=dict(color="#243b53"),
                title_font_color="#102a43"
            )

            st.plotly_chart(fig, width="stretch")

            st.caption(
                "Forecasts are model estimates, "
                "not guaranteed future prices."
            )

        except Exception as e:

            st.error(f"Prediction error: {e}")


# ==========================================================
# 13. PAGE: HISTORICAL ANALYSIS
# ==========================================================

elif page == "Historical Analysis":

    page_header(
        "Historical Price Analysis",
        "Explore and compare rice price estimates "
        "across Sri Lankan markets."
    )

    default_markets = (
        ["Colombo City"]
        if "Colombo City" in market_list
        else market_list[:1]
    )

    selected_markets = st.multiselect(
        "Choose Markets",
        market_list,
        default=default_markets
    )

    if selected_markets:

        filtered = history[
            history["mkt_name"].isin(selected_markets)
        ]

        fig = px.line(
            filtered,
            x="price_date",
            y="c_rice_various",
            color="mkt_name",
            title="Rice Price Trends by Market",
            labels={
                "price_date": "Date",
                "c_rice_various": "Estimated Price (LKR)",
                "mkt_name": "Market"
            }
        )

        fig.update_layout(
            template="plotly_white",
            height=520,
            font=dict(color="#243b53"),
            title_font_color="#102a43"
        )

        st.plotly_chart(fig, width="stretch")

        section("Market Statistics")

        summary = (
            filtered.groupby("mkt_name")["c_rice_various"]
            .agg(["mean", "min", "max", "std"])
            .reset_index()
        )

        summary.columns = [
            "Market",
            "Average Estimate",
            "Minimum",
            "Maximum",
            "Standard Deviation"
        ]

        st.dataframe(
            summary.round(2),
            width="stretch",
            hide_index=True
        )

    else:

        st.warning("Please select at least one market.")


# ==========================================================
# 14. PAGE: MODEL PERFORMANCE
# ==========================================================

elif page == "Model Performance":

    page_header(
        "Model Performance",
        "Compare the performance of all implemented models."
    )

    section("Model Evaluation Results")

    st.dataframe(
        metrics.round(4),
        width="stretch",
        hide_index=True
    )

    if {"Model", "MAE"}.issubset(metrics.columns):

        section("Mean Absolute Error")

        fig = px.bar(
            metrics.sort_values("MAE"),
            x="Model",
            y="MAE",
            text_auto=".2f",
            title="MAE Comparison - Lower is Better"
        )

        fig.update_layout(
            template="plotly_white",
            height=430,
            font=dict(color="#243b53"),
            title_font_color="#102a43"
        )

        st.plotly_chart(fig, width="stretch")

    if {"Model", "RMSE"}.issubset(metrics.columns):

        section("Root Mean Squared Error")

        fig = px.bar(
            metrics.sort_values("RMSE"),
            x="Model",
            y="RMSE",
            text_auto=".2f",
            title="RMSE Comparison - Lower is Better"
        )

        fig.update_layout(
            template="plotly_white",
            height=430,
            font=dict(color="#243b53"),
            title_font_color="#102a43"
        )

        st.plotly_chart(fig, width="stretch")

    if {"Model", "R2"}.issubset(metrics.columns):

        section("R² Score")

        fig = px.bar(
            metrics.sort_values("R2", ascending=False),
            x="Model",
            y="R2",
            text_auto=".2f",
            title="R² Comparison - Higher is Better"
        )

        fig.update_layout(
            template="plotly_white",
            height=430,
            font=dict(color="#243b53"),
            title_font_color="#102a43"
        )

        st.plotly_chart(fig, width="stretch")

    st.info(
        "MAE and RMSE measure prediction errors. "
        "Lower values are better. Higher R² values indicate "
        "better explanatory performance. Negative R² indicates "
        "performance worse than predicting the test-set mean."
    )

    st.warning(
        "Evaluation results use chronological rolling "
        "one-month-ahead testing. They do not guarantee "
        "long-term forecast accuracy."
    )


# ==========================================================
# 15. PAGE: FEATURE IMPORTANCE
# ==========================================================

elif page == "Feature Importance":

    page_header(
        "Feature Importance",
        "Understand which features contribute "
        "to Random Forest predictions."
    )

    try:

        rf_estimator = model.named_steps["regressor"]
        preprocessor = model.named_steps["preprocessor"]

        importance = rf_estimator.feature_importances_
        feature_names = preprocessor.get_feature_names_out()

        if len(feature_names) != len(importance):

            raise ValueError(
                "Feature names do not match model importance values."
            )

        importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importance
        }).sort_values("Importance", ascending=False)

        section("Top 15 Important Features")

        fig = px.bar(
            importance_df.head(15).sort_values("Importance"),
            x="Importance",
            y="Feature",
            orientation="h",
            title="Random Forest Feature Importance"
        )

        fig.update_layout(
            template="plotly_white",
            height=550,
            font=dict(color="#243b53"),
            title_font_color="#102a43"
        )

        st.plotly_chart(fig, width="stretch")

        section("Feature Importance Values")

        st.dataframe(
            importance_df.round(5),
            width="stretch",
            hide_index=True
        )

        st.info(
            "Feature importance describes the contribution "
            "of input features to the fitted model. "
            "It does not establish causation."
        )

    except Exception as e:

        st.error(f"Could not load feature importance: {e}")


# ==========================================================
# 16. PAGE: DATA EXPLORER
# ==========================================================

elif page == "Data Explorer":

    page_header(
        "Data Explorer",
        "Explore and download the historical rice price dataset."
    )

    c1, c2, c3 = st.columns(3)

    c1.metric("Total Records", f"{number_records:,}")
    c2.metric("Markets", number_markets)
    c3.metric("Latest Data", latest_date.strftime("%B %Y"))

    # FILTER DATASET

    section("Filter Dataset")

    selected_market = st.selectbox(
        "Select Market",
        ["All Markets"] + market_list
    )

    if selected_market == "All Markets":

        filtered = history.copy()

    else:

        filtered = history[
            history["mkt_name"] == selected_market
        ].copy()

    # DATA PREVIEW

    section("Dataset Preview")

    st.dataframe(
        filtered.sort_values(
            "price_date",
            ascending=False
        ),
        width="stretch",
        hide_index=True
    )

    # STATISTICS

    section("Descriptive Statistics")

    st.dataframe(
        filtered["c_rice_various"].describe().to_frame().round(2),
        width="stretch"
    )

    # FILTERED DATASET DOWNLOAD

    section("Download Filtered Dataset")

    csv = filtered.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Filtered Dataset",
        data=csv,
        file_name="AgriVanta_AI_filtered_data.csv",
        mime="text/csv",
        width="stretch"
    )

    # ORIGINAL DATASET DOWNLOAD

    section("Download Original World Bank Dataset")

    st.write(
        "Download the complete original dataset used "
        "for this project."
    )

    show_original_dataset_download()


# ==========================================================
# 17. FOOTER
# ==========================================================

st.markdown("---")

st.markdown(
    '<div class="footer">'
    'AgriVanta AI | Sri Lankan Food Price Intelligence<br>'
    'Academic Machine Learning Project | Data Source: World Bank'
    '</div>',
    unsafe_allow_html=True
)
