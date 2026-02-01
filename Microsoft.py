import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from prophet import Prophet
from prophet.plot import plot_plotly

st.set_page_config(layout="wide")
st.title("📈 Microsft (MSFT) Stock Price Predictor with Prophet")

# --- 1. Get Stock Data ---
@st.cache_data
def get_stock_data(years=5):
    ticker_symbol = "MSFT"
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years * 365)

    data = yf.download(ticker_symbol, start=start_date, end=end_date)
    return data

# --- 2. Prepare Data ---
@st.cache_data
def prepare_data(data):
    df = data["Close"].reset_index()
    df.columns = ["ds", "y"]
    df["ds"] = pd.to_datetime(df["ds"])
    return df

# --- 3. Train Model ---
@st.cache_resource
def train_model(df):
    model = Prophet()
    model.fit(df)
    return model

# --- App Logic ---
if st.button("Analyze and Predict Microsoft Stock"):
    with st.spinner("Fetching data and training model..."):
        raw_data = get_stock_data(years=5)
        df = prepare_data(raw_data)

        st.subheader("Historical Data (Last 5 Years)")
        st.dataframe(df.tail())

        model = train_model(df)

        future = model.make_future_dataframe(periods=360)
        forecast = model.predict(future)

        st.subheader("Forecast (Next 360 Days)")
        st.dataframe(forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail())

        st.subheader("📊 Forecast Plot")
        fig1 = plot_plotly(model, forecast)
        st.plotly_chart(fig1, use_container_width=True)

        st.subheader("📉 Decomposition (Trend & Seasonality)")
        fig2 = model.plot_components(forecast)
        st.pyplot(fig2)

        st.success("Prediction completed successfully ✅")
