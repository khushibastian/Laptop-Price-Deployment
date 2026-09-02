"""
train_model.py
Trains Linear Regression, Random Forest, Extra Trees, and Gradient Boosting
on the laptop price dataset, prints R2/MAE for each, and saves the
Gradient Boosting pipeline (best performer) as pipe.pkl for the Streamlit app.

Run this once (locally or in Colab) before running app.py.
"""

import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score, mean_absolute_error

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor

# ---------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------
df = pd.read_csv("Cleaned_data.csv")
if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])

X = df.drop(columns=["Price"])
y = df["Price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

categorical_features = ["Company", "TypeName", "Cpu", "Gpu", "OpSys"]

preprocessor = ColumnTransformer(
    transformers=[
        ("col_tnf", OneHotEncoder(sparse_output=False, handle_unknown="ignore"), categorical_features)
    ],
    remainder="passthrough",
)

# ---------------------------------------------------------
# 2. Define models to compare
# ---------------------------------------------------------
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42),
    "Extra Trees": ExtraTreesRegressor(n_estimators=200, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42),
}

best_name, best_pipe, best_r2 = None, None, -float("inf")

for name, model in models.items():
    pipe = Pipeline([("preprocessor", preprocessor), ("model", model)])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"{name:20s} | R2: {r2:.4f} | MAE: {mae:,.2f}")

    if r2 > best_r2:
        best_name, best_pipe, best_r2 = name, pipe, r2

print(f"\nBest model: {best_name} (R2 = {best_r2:.4f})")

# ---------------------------------------------------------
# 3. Save the best pipeline + a reference dataframe (for dropdown options)
# ---------------------------------------------------------
with open("pipe.pkl", "wb") as f:
    pickle.dump(best_pipe, f)

with open("df.pkl", "wb") as f:
    pickle.dump(df, f)

print("Saved pipe.pkl and df.pkl")