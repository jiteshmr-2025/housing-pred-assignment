# 🏠 Housing Rental Price Prediction (KL & Selangor)

**Date:** October 2023

**Project Type:** Machine Learning / Regression Analysis

**Tools:** Python, Pandas, Scikit-Learn, Seaborn

---

## 1. Executive Summary

The goal of this analysis was to build a machine learning model to estimate the monthly rental price of apartments in Kuala Lumpur and Selangor. Using a dataset of **~9,400 listings**, we developed a **Random Forest Regressor** model that predicts rental prices with **71% accuracy ()**.

The model identified that **Property Size**, **Luxury Facilities**, and **Location** are the strongest drivers of rental prices in this market.

---

## 2. Problem Statement

Real estate pricing is often subjective. Landlords and tenants struggle to determine a "fair market price" due to:

* Complex combinations of features (e.g., Is a small unit in a rich area worth more than a big unit in a poor area?).
* Skewed market data (Luxury penthouses skewing averages).
* Lack of transparency in facility valuation (How much is a pool worth?).

This project solves this by creating a data-driven **Price Prediction Engine**.

---

## 3. Data Overview

* **Source:** `mudah-apartment-kl-selangor.csv`
* **Size:** 9,424 rows, 18 columns.
* **Key Features:**
* `rent`: Monthly rental price (RM) - **Target Variable**.
* `size`: Floor area in square feet.
* `rooms`, `bathroom`, `parking`: Structural features.
* `district`: Location (59 unique districts).
* `property_type`: Condominium, Flat, Serviced Residence, etc.
* `facilities`: Binary flags (Pool, Gym, Air-Con, Near Rail, etc.).



---

## 4. Methodology & Workflow

### Phase 1: Data Cleaning

* **Handling Nulls:** Removed rows with missing critical values (`size`, `rent`).
* **Outlier Removal:** Filtered illogical data (e.g., Size < 200 sq.ft or > 6000 sq.ft).
* **Dimensionality Reduction:** Dropped `region` (Zero Variance) and `prop_name` (High Cardinality - 1,285 unique values).

### Phase 2: Exploratory Data Analysis (EDA)

* **Univariate:** Identified that `rent` is highly right-skewed (median RM 1,600, max RM 16,800).
* **Bivariate:**
* Strong positive correlation between `size` and `rent`.
* "Fully Furnished" units command a noticeable premium over "Partially Furnished".


* **Multivariate:** Confirmed that *Serviced Residences* have a steeper price-per-sqft slope than *Apartments*.

### Phase 3: Feature Engineering

To improve model performance, new features were created:

1. **`rent_log`**: Log-transformation of the target variable to normalize distribution.
2. **`luxury_score`**: A composite score (0-6) summing up facilities (Pool + Gym + AC + Rail + Internet + Cooking).
3. **`size_per_room`**: Calculated as `size / rooms` to measure spaciousness.
4. **`district_grouped`**: Grouped the 59 districts into "Top 15" + "Other" to reduce noise.

### Phase 4: Model Selection & Training

We employed a **Champion vs. Challenger** approach:

* **Baseline Model:** Linear Regression.
* **Challenger Model:** Random Forest Regressor.

Both models were trained on an 80/20 Train-Test split.

---

## 5. Model Results

| Model | RMSE (Error in RM) |  Score (Accuracy) | Verdict |
| --- | --- | --- | --- |
| **Linear Regression** | RM 640.12 | 0.65 (65%) | Underfitting |
| **Random Forest** | **RM 573.57** | **0.71 (71%)** | **🏆 Winner** |

**Why Random Forest Won:**
Housing data contains non-linear relationships (e.g., "Near Rail" might increase price in the suburbs but matters less in the city center). Random Forest captures these complex interactions better than a straight line.

---

## 6. Key Insights

According to the Feature Importance analysis, the top drivers of rental price are:

1. **Size (sq.ft.):** The #1 predictor. Space is the primary value driver.
2. **Luxury Score:** The presence of amenities (Pool, Gym, AC) is the second most important factor.
3. **District:** Being in *Mont Kiara* or *KL City Centre* significantly shifts the base price.
4. **Furnishing:** A fully furnished unit adds a tangible premium.

---

## 7. How to Use the Model

You can use the trained model to predict rent for new listings.

### Dependencies

```python
import pandas as pd
import numpy as np
import joblib

```

### Prediction Code Snippet

```python
# Load Model
model = joblib.load('final_housing_model.pkl')

# Example Input: 1200 sqft Condo in Cheras
# (Note: Data must be preprocessed to match training columns)
prediction = model.predict(input_data)
price_rm = np.expm1(prediction) # Convert log back to RM

print(f"Recommended Rent: RM {price_rm}")

```

---

## 8. Future Improvements

To reduce the error margin (RMSE) further, future iterations could:

1. **Scrape Qualitative Data:** Incorporate text data like "Renovated," "Corner Unit," or "High Floor."
2. **Remove Extreme Luxury:** Creating a separate model for "Ultra-Luxury" units (Rent > RM 8,000) might improve accuracy for the mass market.
3. **Geospatial Analysis:** Use exact Latitude/Longitude to measure distance to KLCC specifically.

---
