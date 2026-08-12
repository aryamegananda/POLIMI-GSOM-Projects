# 0. Import
import streamlit as st
import pandas as pd
import yfinance as yf
import requests
from bs4 import BeautifulSoup as bs

# 1. Title
st.set_page_config(layout="wide", page_title="Stock Buddy")
st.title("STOCK BUDDY")


# 2. Variables
tickers = ["NVDA", "AAPL", "MSFT", "AMZN", "GOOG", "META", "SPCX", "MU", "TSLA", "AMD", "ASML", "TSM", "INTC"]
portofolio = ["MU", "GOOG", "TSM", "ASML"]
page = st.sidebar.radio("Navigate", ["Home", "Build Portofolio", "Fundamental", "Financials", "Technical", "News"])
all_data = {}
all_info = {}
all_scores = {}


# 3. Load
@st.cache_data
def load_all_data(tickers):
    all_data = {}
    all_info = {}

    for t in tickers:
        try:
            df = yf.download(t, period="6mo")
            df.columns = df.columns.droplevel("Ticker")

            # Technical Indicators
            df["SMA_20"] = df["Close"].rolling(window=20).mean()
            df["SMA_50"] = df["Close"].rolling(window=50).mean()
            delta = df["Close"].diff()
            avg_gain = delta.clip(lower=0).rolling(window=14).mean()
            avg_loss = delta.clip(upper=0).abs().rolling(window=14).mean()
            rs = avg_gain / avg_loss
            df["RSI"] = 100 - (100/(1+rs))
            df["Volatility"] = df["Close"].pct_change().rolling(window=20).std()
            df["Daily_Return"] = df["Close"].pct_change()

            all_data[t] = df
            all_info[t] = yf.Ticker(t).info

        except:
            pass
    return all_data, all_info

all_data, all_info = load_all_data(tickers)


# 4. Score functions
def calc_fundamental_score(info):
    score = 0

    pe = info.get("trailingPE", None)
    if pe is not None and pe < 20:
        score += 1
    elif pe is not None and pe > 40:
        score -= 1
    roe = info.get("returnOnEquity", None)
    if roe is not None and roe > 0.02:
        score += 1
    de = info.get("debtToEquity", None)
    if de is not None and de > 100:
        score -= 1
    return score

def calc_technical_score(df):
    score = 0
    latest = df.iloc[-1]
    if latest["RSI"] < 30:
        score += 2
    elif latest["RSI"] > 70:
        score -= 2
    if latest["Close"] > latest["SMA_20"]:
        score += 1
    else:
        score -= 1
    if latest["Close"] > latest["SMA_50"]:
        score += 1
    else:
        score -= 1
    if latest["Volatility"] < 0.03:
        score += 1
    elif latest["Volatility"] > 0.06:
        score -= 1
    return score

def get_news(ticker):
    url = f"https://finviz.com/quote.ashx?t={ticker}"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}
    response = requests.get(url, headers=headers)
    soup = bs(response.text, "html.parser")

    headlines = soup.find_all("a", class_= "tab-link-news")

    news = []
    for h in headlines[:10]:
        news.append({
            "title":h.text.strip(),
            "link":h.get("href")
        })

    return news


# 5. Sidebar
if page == "Home":
    st.subheader("Portfolio Overview")
    
    # Calculate portfolio score
    porto_score = 0
    porto_count = 0
    for t in portofolio:
        if t in all_data:
            tech = calc_technical_score(all_data[t])
            fund = calc_fundamental_score(all_info[t])
            porto_score += tech + fund
            porto_count += 1
    
    if porto_count > 0:
        avg_score = porto_score / porto_count
        st.metric("Portfolio Average Score", f"{avg_score:.1f} / 7")
    
    # Display individual stocks
    for t in portofolio:
        if t in all_data:
            tech = calc_technical_score(all_data[t])
            fund = calc_fundamental_score(all_info[t])
            total = tech + fund
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric(t, f"${all_data[t]['Close'].iloc[-1]:.2f}")
            with col2:
                st.metric("Technical", f"{tech}")
            with col3:
                st.metric("Fundamental", f"{fund}")
            with col4:
                st.metric("Total", f"{total}")

elif page == "Build Portofolio":
    st.subheader("Build the Portofolio Here")
    st.write("coming soon.")

elif page == "Fundamental":
    selected = st.sidebar.selectbox("Select a stock", tickers)

    st.subheader(f"{selected} - Key Ratios")

    info = all_info[selected]
    fund_scores = calc_fundamental_score(info)
    st.metric("Fundamental Score", f"{fund_scores} /3")

    col1, col2 = st.columns(2)

    with col1:
        pe = info.get('trailingPE', None)
        st.metric("P/E Ratio", f"{pe:.1f}" if pe else "N/A")
        
        pb = info.get('priceToBook', None)
        st.metric("P/B Ratio", f"{pb:.1f}" if pb else "N/A")
        
        roe_val = info.get('returnOnEquity', None)
        st.metric("ROE", f"{roe_val:.1%}" if roe_val else "N/A")
        
        de_val = info.get('debtToEquity', None)
        st.metric("Debt/Equity", f"{de_val:.1f}" if de_val else "N/A")

    with col2:
        gm = info.get('grossMargins', None)
        st.metric("Gross Margin", f"{gm:.1%}" if gm else "N/A")
        
        pm = info.get('profitMargins', None)
        st.metric("Profit Margin", f"{pm:.1%}" if pm else "N/A")
        
        roa_val = info.get('returnOnAssets', None)
        st.metric("ROA", f"{roa_val:.1%}" if roa_val else "N/A")
        
        dy = info.get('dividendYield', None)
        st.metric("Dividend Yield", f"{dy:.2f}" if dy else "N/A")

elif page == "Financials":
    selected = st.sidebar.selectbox("Select a stock", tickers)
    st.subheader(f"{selected} - Financials")

elif page == "Technical":
    selected = st.sidebar.selectbox("Select a stock", tickers)
    st.subheader(f"{selected} - Key Technical Analysis")

    df = all_data[selected]
    tech_score = calc_technical_score(df)

    st.metric("Technical Score", f"{tech_score} /5")

    # Table
    latest = df.iloc[-1]
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Close", f"${latest['Close']:.2f}")
    with col2:
        st.metric("SMA 20", f"${latest['SMA_20']:.2f}")
    with col3:
        st.metric("RSI", f"{latest['RSI']:.1f}")
    with col4:
        st.metric("Volatility", f"{latest['Volatility']:.4f}")

    # Price & Moving Average
    st.subheader("Price & Moving Average")
    st.line_chart(df[["Close", "SMA_20", "SMA_50"]].dropna())

    # RSI
    st.subheader("RSI")
    st.line_chart(df["RSI"].dropna())

    # Volatility
    st.subheader("Volatility")
    st.line_chart(df["Volatility"].dropna())
    

elif page == "News":
    selected = st.sidebar.selectbox("Select a stock", tickers)
    st.subheader(f"{selected} - News")

    news = get_news(selected)

    for article in news:
        st.markdown(f"[{article['title']}]({article['link']})")