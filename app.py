
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objs as go
import ta
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import talib

st.set_page_config(page_title="تحليل الذهب AI", layout="wide")
st.title("🤖📈 تحليل الذهب (XAU/USD) بالذكاء الاصطناعي")

data = yf.download("XAUUSD=X", interval="15m", period="7d")
if data.empty:
    st.error("فشل تحميل البيانات!")
    st.stop()

data['RSI'] = ta.momentum.RSIIndicator(data['Close'], window=14).rsi()
data['MACD'] = ta.trend.MACD(data['Close']).macd()
data['EMA20'] = ta.trend.EMAIndicator(data['Close'], window=20).ema_indicator()
bb = ta.volatility.BollingerBands(data['Close'])
data['Boll_Upper'] = bb.bollinger_hband()
data['Boll_Lower'] = bb.bollinger_lband()
data['ATR'] = ta.volatility.AverageTrueRange(data['High'], data['Low'], data['Close']).average_true_range()
data['Momentum'] = ta.momentum.MomentumIndicator(data['Close'], window=10).momentum()
data['Stochastic'] = ta.momentum.StochasticOscillator(data['High'], data['Low'], data['Close']).stoch()

patterns = ['CDLENGULFING', 'CDLDOJI', 'CDLHAMMER', 'CDLSHOOTINGSTAR',
            'CDLMORNINGSTAR', 'CDLEVENINGSTAR', 'CDLPIERCING', 'CDLDARKCLOUDCOVER']
for pattern in patterns:
    func = getattr(talib, pattern)
    data[pattern] = func(data['Open'], data['High'], data['Low'], data['Close'])

def get_dominant_pattern(row):
    for pattern in patterns:
        if row[pattern] != 0:
            return pattern + ("_Bullish" if row[pattern] > 0 else "_Bearish")
    return ""

data['Candle_Pattern'] = data.apply(get_dominant_pattern, axis=1)

recent = data[-100:]
high = recent['High'].max()
low = recent['Low'].min()
diff = high - low
fib_levels = [high,
              high - 0.236 * diff,
              high - 0.382 * diff,
              high - 0.5 * diff,
              high - 0.618 * diff,
              high - 0.786 * diff,
              low]

features = ['RSI', 'MACD', 'EMA20', 'Momentum', 'ATR', 'Stochastic']
df_model = data[features].dropna()
X = df_model
y = np.where(data.loc[X.index, 'Close'].shift(-1) > data.loc[X.index, 'Close'], 1, 0)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = RandomForestClassifier()
model.fit(X_scaled, y)

latest = X.iloc[-1:]
latest_scaled = scaler.transform(latest)
pred = model.predict(latest_scaled)[0]
direction = "🔼 شراء" if pred == 1 else "🔽 بيع"

close = data['Close'].iloc[-1]
atr = data['ATR'].iloc[-1]
tp = close + atr * 2 if pred == 1 else close - atr * 2
sl = close - atr if pred == 1 else close + atr

st.subheader("📢 التوصية")
st.markdown(f"### الاتجاه المتوقع: {direction}")
st.markdown(f"**TP (الهدف):** {tp:.2f} | **SL (الوقف):** {sl:.2f}")

fig = go.Figure()
fig.add_trace(go.Candlestick(x=data.index, open=data['Open'], high=data['High'],
                             low=data['Low'], close=data['Close'], name='XAU/USD'))
fig.add_trace(go.Scatter(x=data.index, y=data['EMA20'], mode='lines', name='EMA 20'))
fig.add_trace(go.Scatter(x=data.index, y=data['Boll_Upper'], mode='lines', name='Boll Upper'))
fig.add_trace(go.Scatter(x=data.index, y=data['Boll_Lower'], mode='lines', name='Boll Lower'))

for lvl in fib_levels:
    fig.add_hline(y=lvl, line_dash="dash", line_color="orange")

st.plotly_chart(fig, use_container_width=True)

st.subheader("📊 بيانات فنية + شموع")
st.dataframe(data[['RSI', 'MACD', 'EMA20', 'ATR', 'Candle_Pattern']].tail(15))
