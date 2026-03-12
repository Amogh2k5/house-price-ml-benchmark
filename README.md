# 🏠 House Price Prediction

A machine learning project that predicts house prices using multiple regression models. Built with Python and scikit-learn, comparing the performance of Linear Regression, Random Forest, Gradient Boosting, and XGBoost.

---

## 📁 Project Structure

```
house/
├── Housing.csv                          # Dataset
├── linear_regression_house_price.py     # Linear Regression model
├── house_price_random_forest.py         # Random Forest model
├── house_price_gradient_boosting.py     # Gradient Boosting model
├── house_price_xgboost.py               # XGBoost model
└── README.md
```

---

## 📊 Dataset

The dataset (`Housing.csv`) contains features such as:
- Area, number of bedrooms and bathrooms
- Parking, stories, and furnishing status
- Binary features: mainroad, guestroom, basement, hot water heating, air conditioning

**Target variable:** `price`

---

## 🤖 Models Used

| Model | Description |
|---|---|
| Linear Regression | Baseline model |
| Random Forest | Ensemble of decision trees |
| Gradient Boosting | Sequential boosting approach |
| XGBoost | Optimized gradient boosting |

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install pandas numpy scikit-learn xgboost matplotlib
```

### Run a model

```bash
python linear_regression_house_price.py
python house_price_random_forest.py
python house_price_gradient_boosting.py
python house_price_xgboost.py
```

---

## 🛠️ Tech Stack

- **Python**
- **pandas** — data manipulation
- **scikit-learn** — ML models and preprocessing
- **XGBoost** — gradient boosting
- **matplotlib / seaborn** — visualization

---

## 📈 Results

> Update this section with your model scores after running the scripts.

| Model | R² Score | RMSE |
|---|---|---|
| Linear Regression | — | — |
| Random Forest | — | — |
| Gradient Boosting | — | — |
| XGBoost | — | — |

---

## 🔮 Future Improvements

- [ ] Hyperparameter tuning with GridSearchCV
- [ ] Cross-validation for more reliable evaluation
- [ ] Feature importance visualization
- [ ] Streamlit web app for live predictions

---

## 👤 Author

**Amogh** — [github.com/Amogh2k5](https://github.com/Amogh2k5)
