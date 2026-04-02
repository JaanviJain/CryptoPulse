# 🚀 Crypto Pulse

**Self-hosted LLM-powered Crypto Trading Intelligence System**

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Ollama](https://img.shields.io/badge/Ollama-Llama%203.2-orange.svg)](https://ollama.ai)
[![XGBoost](https://img.shields.io/badge/XGBoost-ML-red.svg)](https://xgboost.ai)



## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Running the Application](#running-the-application)
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
- [API Keys Setup](#api-keys-setup)
- [Dashboard Views](#dashboard-views)
- [Performance Metrics](#performance-metrics)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## 📖 Overview

The **Crypto Pulse** is a complete, self-hosted cryptocurrency trading intelligence system that combines:

- **LLM Sentiment Analysis** (Ollama + Llama 3.2) - Analyzes news articles in real-time
- **XGBoost Price Prediction** - Predicts next 1-hour price direction with >55% accuracy
- **On-Chain Monitoring** - Tracks whale transactions >$1M and exchange flows
- **Real-Time Dashboard** - CLI (Rich) and Web (Streamlit) interfaces

**Why this matters:**
- Professional sentiment APIs cost ₹5L+/month - this is FREE
- Runs entirely on your laptop (no cloud costs)
- Combines sentiment + technicals + on-chain data for better signals

## ✨ Features

### ✅ Completed Features (MVP)

| Feature | Status | Description |
|---------|--------|-------------|
| Data Ingestion | ✅ | Binance API for OHLCV price data (15-min candles) |
| Sentiment Analysis | ✅ | Llama 3.2 via Ollama for news sentiment scoring |
| Price Prediction | ✅ | XGBoost model with 55-60% directional accuracy |
| On-Chain Data | ✅ | Whale alerts + exchange flow tracking |
| Signal Generator | ✅ | BUY/SELL/HOLD with confidence & reasoning |
| Backtesting | ✅ | 30-day P&L, Win rate, Sharpe ratio |
| CLI Dashboard | ✅ | Rich library terminal interface |
| Web Dashboard | ✅ | Streamlit interactive interface |

