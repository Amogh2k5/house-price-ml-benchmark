# linear_regression_house_price.py

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

import matplotlib.pyplot as plt

print("🔹 Script started")

# ---------------------------
# LOAD DATA
# ---------------------------
try:
    df = pd.read_csv("Housing.csv")  # Make sure file is in same folder
    print("✅ Dataset loaded:", df.shape)
    print(df.head())
except Exception as e:
    print("❌ Error while loading Housing.csv:")
    print(e)
    print("\n👉 Check that Housing.csv is in the same folder as this .py file.")
    raise SystemExit

# ---------------------------
# FEATURES & TARGET
# ---------------------------
print("\n🔹 Preparing features and target...")
X = df.drop("price", axis=1)
y = df["price"]

# ---------------------------
# IDENTIFY COLUMN TYPES
# ---------------------------
numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()

print("Numeric columns:", numeric_cols)
print("Categorical columns:", categorical_cols)

# ---------------------------
# PREPROCESSING
# ---------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(drop="first"), categorical_cols),
    ]
)

# ---------------------------
# CREATE PIPELINE (Preprocessing → Linear Regression)
# ---------------------------
model = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("regressor", LinearRegression())
])

# ---------------------------
# TRAIN-TEST SPLIT
# ---------------------------
print("\n🔹 Splitting data into train and test...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------------------
# TRAIN MODEL
# ---------------------------
print("🔹 Training model...")
model.fit(X_train, y_train)

# ---------------------------
# PREDICT
# ---------------------------
print("🔹 Making predictions...")
y_pred = model.predict(X_test)

# ---------------------------
# EVALUATION
# ---------------------------
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("\n✅ MODEL PERFORMANCE (Linear Regression)")
print("R² Score :", r2)
print("RMSE     :", rmse)

# ---------------------------
# PLOT: Actual vs Predicted
# ---------------------------
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted House Prices (Linear Regression)")
plt.grid(True)
plt.tight_layout()
plt.show()

# ---------------------------
# SAMPLE PREDICTION
# ---------------------------
sample = X.iloc[[0]]
predicted_price = model.predict(sample)[0]

print("\n🔹 Sample Input:", sample.to_dict(orient="records")[0])
print("🔹 Predicted Price:", predicted_price)

print("\n🎉 Done!")
