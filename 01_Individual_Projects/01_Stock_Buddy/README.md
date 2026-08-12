# Stock Buddy — Portfolio Screening & Signal Engine

A stock screening dashboard built to help me evaluate and track my investment portfolio. The app pulls real-time market data, calculates technical and fundamental metrics, scores each stock on a rule-based signal engine, and displays everything in an interactive Streamlit dashboard.

## Features

- **Technical Analysis** — Moving averages (SMA 20/50), RSI, volatility, with interactive charts
- **Fundamental Analysis** — P/E, P/B, ROE, ROA, profit margins, debt-to-equity, and more
- **Financial Statements** — Quarterly income statement with Y/Y change
- **Signal Scoring** — Rule-based scoring engine that rates each stock from -7 to +7
- **Portfolio Tracker** — Monitor portfolio health with an aggregated score
- **News Feed** — Fetch current news for selected stocks

## Tech Stack

- **Python** — Core language
- **yfinance** — Market data and financial statements
- **pandas** — Data manipulation and indicator calculations
- **Streamlit** — Interactive dashboard