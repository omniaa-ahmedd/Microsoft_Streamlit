import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from prophet import Prophet
from prophet.plot import plot_plotly

st.set_page_config(layout="wide")
st.title("📈 Microsft (MSFT) Stock Price Predictor with Prophet")

@st.cache_data
def get_stock_data(years=5):
    ticker_symbol = "AAPL"
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years * 365)

    data = yf.download(ticker_symbol, start=start_date, end=end_date)

if data.empty:
        return None
return data

    @st.cache_data
def prepare_data(data):
    df = data["Close"].reset_index()
    df.columns = ["ds", "y"]

    df["ds"] = pd.to_datetime(df["ds"])
    df["y"] = pd.to_numeric(df["y"], errors="coerce")

    # 🔴 Drop NaN values
    df = df.dropna()

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

        # 🔴 CHECK 1: Did yfinance return data?
        if raw_data is None or raw_data.empty:
            st.error("❌ Failed to download stock data. Please try again later.")
            st.stop()

        df = prepare_data(raw_data)

        # 🔴 CHECK 2: Does Prophet have enough data?
        if df.shape[0] < 2:
            st.error("❌ Not enough valid data to train the model.")
            st.stop()

    
        st.subheader("Historical Data (Last 5 Years)")
        st.dataframe(df.tail())


        model = train_model(df)

        future = model.make_future_dataframe(periods=360)
        forecast = model.predict(future)

        st.subheader("Forecast (Next 360 Days)")
        st.dataframe(
            forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail()
        )


        st.subheader("📊 Forecast Plot")
        fig1 = plot_plotly(model, forecast)
        st.plotly_chart(fig1, use_container_width=True)

        st.subheader("📉 Decomposition (Trend & Seasonality)")
        st.pyplot(model.plot_components(forecast))

        st.success("Prediction completed successfully ✅")
    
