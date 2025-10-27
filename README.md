# 📊 Cryptocurrency Price Forecasting using Kalman Filter & Linear Regression

This project implements **Kalman Filtering** and **Linear Regression** techniques to forecast cryptocurrency prices (BTC & ETH).  
It explores the use of **probabilistic state estimation** (Kalman Filter) and **supervised regression** for short-term price prediction, model evaluation, and performance comparison.

---

## 🚀 Features

- 📈 **Kalman Filter Smoothing:** Applies the Kalman Filter to denoise BTC and ETH price signals.  
- 🧮 **Forecasting:** Predicts future price trends for different horizons (10, 50, 100, 200 days).  
- 🤖 **Linear Regression Model:** Trains a regression model using scaled price features.  
- 📊 **Performance Metrics:** Evaluates forecasts with R², MAPE, RMSE, and MAE.  
- 🧠 **Modular Code:** Easily adaptable for other time series (e.g., stocks, forex).

---

## 🧰 Technologies Used

- **Programming Language:** Python  
- **Libraries:**  
  - `numpy`, `pandas`, `matplotlib`, `seaborn`  
  - `pandas_datareader` (for fetching data)  
  - `pmdarima`, `pandas_market_calendars`  
  - `pykalman` (Kalman Filter)  
  - `sklearn` (Regression, Metrics, Scaling)

---

## 🧩 Project Structure

