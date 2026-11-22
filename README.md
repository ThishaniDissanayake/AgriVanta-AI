# 🌾 AgriVanta AI
### Intelligent Food Commodity Price Prediction and Market Analysis in Sri Lanka Using Ensemble and Hybrid Machine Learning

**Agricultural Price Intelligence — Predict • Analyse • Understand**

## 1. Project Overview

AgriVanta AI is a machine learning-based food commodity price prediction and market analysis application developed to support the understanding of rice price trends across different markets in Sri Lanka.

The project uses historical monthly food price estimates from the World Bank to analyse price patterns, evaluate machine learning and statistical forecasting techniques, and generate future price predictions.

The application provides an interactive dashboard built with Streamlit, allowing users to explore historical prices, compare model performance, examine feature importance, and generate market-specific price forecasts.

## 2. Project Objectives

- Analyse historical rice price trends across Sri Lankan markets.
- Develop and evaluate machine learning and time-series forecasting models.
- Compare predictive performance using MAE, RMSE and R².
- Generate future rice price predictions for selected markets.
- Provide an interactive dashboard for market analysis and data exploration.
- Demonstrate how data-driven forecasting can support agricultural price intelligence.

## 3. Dataset Description

The project uses the **Monthly Food Price Estimates by Product and Market** dataset for Sri Lanka, obtained from the World Bank Microdata Library.

| Dataset information | Description |
|---|---|
| Source | World Bank Microdata Library |
| Country | Sri Lanka |
| Data period | January 2007 – November 2026 |
| Number of records | 10,516 |
| Individual markets | 43 |
| Districts | 26 |
| Target variable | `c_rice_various` |
| Currency | Sri Lankan Rupees (LKR) |

The target variable represents the World Bank's modeled monthly rice price estimate.

**Dataset source:** https://microdata.worldbank.org/catalog/6169

**Dataset DOI:** https://doi.org/10.48529/2ZH0-JF55

## 4. Machine Learning and Forecasting Models

The project explores and compares the following approaches:

- Random Forest Regressor
- XGBoost Regressor
- LightGBM Regressor
- CatBoost Regressor
- Stacking Ensemble
- ARIMA (1,1,1)
- Hybrid Random Forest + ARIMA

A chronological train-test split is used to respect the time-series nature of the data.

### Model Evaluation

The models are evaluated using:

- **MAE (Mean Absolute Error):** Measures the average absolute prediction error.
- **RMSE (Root Mean Squared Error):** Measures prediction error while penalising larger errors.
- **R² Score:** Measures how well the model explains variation in the target values.

### Model Comparison Results

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Random Forest | 7.4735 | 9.4045 | 0.1979 |
| Tuned Random Forest | 7.0069 | 8.8585 | 0.2883 |
| XGBoost | 12.0185 | 15.1291 | -1.0759 |
| LightGBM | 10.4455 | 13.2750 | -0.5983 |
| CatBoost | 13.9544 | 17.0704 | -1.6428 |
| Simple Average Ensemble | 10.6161 | 13.3047 | -0.6054 |
| Stacking Ensemble | 71.1384 | 75.9178 | -51.2713 |
| ARIMA (1,1,1) | **4.2482** | **6.4808** | **0.6191** |
| Hybrid RF + ARIMA | 4.8428 | 6.5573 | 0.6100 |

**Best-performing model:** ARIMA (1,1,1), based on the evaluated test-set MAE and RMSE.

The results demonstrate that a traditional statistical time-series model can outperform the tested machine learning and ensemble approaches for this dataset.

*Note: Evaluation uses rolling one-month-ahead predictions, with actual previous test observations available when forecasting the next test month. Results should not be interpreted as fixed long-horizon forecasting performance.*

## 5. Dashboard Features

AgriVanta AI provides six main dashboard sections:

### 1. Overview
- Project introduction and key information.
- Best-performing model summary.
- Historical rice price trends.
- Original dataset download.

### 2. Price Prediction
- Select a Sri Lankan market.
- Generate next-month rice price predictions.
- Compare Random Forest, ARIMA and Hybrid predictions.
- View prediction results and charts.

### 3. Historical Analysis
- Explore historical price trends.
- Compare multiple markets.
- Analyse market-level price statistics.

### 4. Model Performance
- Compare machine learning and forecasting models.
- View MAE, RMSE and R² results.
- Explore model evaluation charts.

### 5. Feature Importance
- Examine the relative importance of input features used by the Random Forest model.

### 6. Data Explorer
- Explore the historical dataset.
- Filter data by market.
- View descriptive statistics.
- Download filtered data.

## 6. Technologies Used

- **Programming Language:** Python
- **Data Processing:** Pandas, NumPy
- **Machine Learning:** Scikit-learn, XGBoost, LightGBM, CatBoost
- **Time-Series Forecasting:** Statsmodels (ARIMA)
- **Visualization:** Plotly
- **Dashboard Development:** Streamlit
- **Model Persistence:** Joblib
- **Version Control:** Git and GitHub
- **Development Environment:** Jupyter Notebook and Visual Studio Code

## 7. Project Structure

```text
AgriVanta-AI/
│
├── app.py
├── requirements.txt
├── README.md
│
├── LankaPrice_AI_Ensemble_Learning.ipynb
│
├── lankaprice_data/
│   └── LKA_RTFP_mkt_2007_2026-11-17.csv
│
├── lankaprice_models/
│   ├── final_random_forest.pkl
│   ├── market_history.csv
│   └── model_comparison.csv
│
└── .gitignore
```

## 8. Installation and Setup

### Prerequisites

- Python 3.12 recommended
- Git
- A GitHub account

### Step 1: Clone the Repository

```bash
git clone https://github.com/ThishaniDissanayake/AgriVanta-AI.git
```

### Step 2: Navigate to the Project Directory

```bash
cd AgriVanta-AI
```

### Step 3: Create a Virtual Environment

```bash
python -m venv .venv
```

### Step 4: Activate the Virtual Environment

**Windows PowerShell:**

```powershell
.\.venv\Scripts\Activate.ps1
```

### Step 5: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 6: Run the Application

```bash
python -m streamlit run app.py
```

The application will open in your web browser.

## 9. Future Improvements

- Explore more advanced time-series forecasting techniques.
- Improve prediction accuracy across individual markets.
- Investigate the impact of economic and seasonal factors.
- Introduce longer-horizon forecasting and uncertainty intervals.
- Improve model performance through further hyperparameter tuning.
- Integrate additional agricultural and economic datasets.
- Deploy the application for wider public access.

## 10. Acknowledgements

### Dataset Acknowledgement

The dataset used in this project was obtained from the World Bank Microdata Library. I gratefully acknowledge the World Bank and the dataset contributors for providing access to the Sri Lankan monthly food price estimates.

**Reference:**

Andrée, B. P. J. (2021). *Monthly food price estimates by product and market* (Version 2026-11-17). World Bank Microdata Library. Sri Lanka dataset, LKA_2021_RTFP_v02_M.

https://doi.org/10.48529/2ZH0-JF55

Dataset licence: Creative Commons Attribution 4.0 International (CC BY 4.0).

### Generative AI Acknowledgement

ChatGPT, developed by OpenAI, was used as a supporting tool during the development of this project. It assisted with Python code development, debugging, machine learning explanations, model evaluation guidance, and Streamlit dashboard design.

All generated suggestions and code were reviewed, adapted and tested as part of the project development process. Responsibility for the project implementation, analysis, interpretation and reported results remains with the project author.

## 11. Author

**Project:** AgriVanta AI  
**Academic Topic:** Intelligent Food Commodity Price Prediction and Market Analysis in Sri Lanka Using Ensemble and Hybrid Machine Learning.

**GitHub Repository:**  
https://github.com/ThishaniDissanayake/AgriVanta-AI

---

*AgriVanta AI — Agricultural Price Intelligence: Predict • Analyse • Understand*
