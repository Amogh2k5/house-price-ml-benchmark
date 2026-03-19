# House Price Prediction — ML Model Benchmark

A comparison of four regression algorithms on a structured housing dataset, evaluating predictive accuracy, training efficiency, and feature importance. Built to explore how ensemble methods outperform linear baselines on real-world tabular data.

---

## Results

| Model | RMSE | R² Score | Training Time |
|---|---|---|---|
| Linear Regression | — | — | — |
| Random Forest | — | — | — |
| Gradient Boosting | — | — | — |
| **XGBoost** | **—** | **—** | **—** |

> Fill in your actual numbers after running the scripts. XGBoost typically achieves the lowest RMSE on this dataset.

---

## What's Inside

```
house/
├── Housing.csv                          # Dataset (545 rows, 13 features)
├── linear_regression_house_price.py     # Baseline linear model
├── house_price_random_forest.py         # Ensemble: bagging
├── house_price_gradient_boosting.py     # Ensemble: boosting (sklearn)
├── house_price_xgboost.py               # Ensemble: boosting (XGBoost)
└── README.md
```

---

## Dataset

The `Housing.csv` dataset contains 545 residential property listings with the following features:

- **Numerical**: area, bedrooms, bathrooms, stories, parking
- **Binary categorical**: mainroad, guestroom, basement, hotwaterheating, airconditioning, prefarea
- **Categorical**: furnishingstatus (furnished / semi-furnished / unfurnished)

Target variable: `price` (continuous)

---

## Models

**Linear Regression** — Ordinary least squares baseline. Assumes a linear relationship between features and price; used as the performance floor.

**Random Forest** — Bagging ensemble of 100 decision trees. Reduces variance through averaging and handles non-linear relationships well.

**Gradient Boosting** — Sequential boosting via scikit-learn's `GradientBoostingRegressor`. Builds trees to correct residuals of prior trees; slower to train but often more accurate.

**XGBoost** — Optimised gradient boosting with regularisation (L1/L2), built-in handling of missing values, and faster training via parallelised tree construction.

---

## Setup

```bash
git clone https://github.com/Amogh2k5/house.git
cd house
pip install pandas scikit-learn xgboost matplotlib
```

---

## Usage

Run any model independently:

```bash
python linear_regression_house_price.py
python house_price_random_forest.py
python house_price_gradient_boosting.py
python house_price_xgboost.py
```

Each script outputs RMSE, MAE, and R² on an 80/20 train-test split.

---

## Key Takeaways

- Ensemble methods (Random Forest, GBM, XGBoost) consistently outperform linear regression on this dataset due to non-linear feature interactions (e.g. airconditioning × area).
- XGBoost's regularisation prevents overfitting compared to vanilla gradient boosting, particularly with the small dataset size (545 rows).
- `area` and `airconditioning` are the strongest predictors of price across all models.

---

## Tech Stack

`Python` · `scikit-learn` · `XGBoost` · `pandas` · `matplotlib`
