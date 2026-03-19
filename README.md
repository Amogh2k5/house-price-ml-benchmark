# House Price Prediction — ML Model Benchmark

A comparison of four regression algorithms on a structured housing dataset, evaluating predictive accuracy, training efficiency, and feature importance. Built to explore how ensemble methods outperform linear baselines on real-world tabular data.

---

## Results

| Model | RMSE | R² Score |
|---|---|---|
| **Gradient Boosting** | **1,286,462** | **0.673** |
| Linear Regression | 1,324,507 | 0.653 |
| XGBoost | 1,364,171 | 0.632 |
| Random Forest | 1,391,066 | 0.617 |

> Evaluated on an 80/20 train-test split. Gradient Boosting achieves the best R² (0.673) and lowest RMSE, outperforming XGBoost — likely because sklearn's conservative learning rate generalises better on this small dataset (545 rows).

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

- **Gradient Boosting wins** (R² 0.673, RMSE 1,286,462), beating XGBoost despite both being boosting methods. sklearn's `GradientBoostingRegressor` uses a slower, more conservative learning rate by default which generalises better on a small dataset like this.
- **Linear Regression ranks second** (R² 0.653), outperforming XGBoost and Random Forest — showing that on 545 rows with largely linear feature relationships, a simple model can beat complex ensembles.
- XGBoost edges out Random Forest (0.632 vs 0.617), with its L1/L2 regularisation reducing variance compared to unregularised bagging.
- `area` and `airconditioning` are the strongest predictors of price across all models.

---

## Tech Stack

`Python` · `scikit-learn` · `XGBoost` · `pandas` · `matplotlib`
