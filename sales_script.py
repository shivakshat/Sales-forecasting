# ============================================================
# SALES FORECASTING FOR A RETAIL CHAIN
# Time Series (optional), Regression, EDA, SQL, Visualization
# ============================================================

# ------------ IMPORTS ------------
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np

# ------------ LOAD DATA ------------
file_path = "dataset.csv"   # Change if needed
df = pd.read_csv(file_path)

print("Dataset Loaded Successfully!")
print(df.head())

# ------------ SQL INTEGRATION ------------
conn = sqlite3.connect("sales_db.sqlite")
df.to_sql("sales_table", conn, if_exists="replace", index=False)

# Example SQL Query
query = """
SELECT *
FROM sales_table
WHERE Sales > (SELECT AVG(Sales) FROM sales_table)
LIMIT 10;
"""
sql_output = pd.read_sql_query(query, conn)
print("\nSQL Output (Top High Sales Rows):")
print(sql_output.head())

# ------------ EDA ------------
print("\nBasic Info:")
print(df.info())

print("\nSummary Statistics:")
print(df.describe())

# Sales distribution
plt.figure(figsize=(7,4))
sns.histplot(df["Sales"], kde=True)
plt.title("Sales Distribution")
plt.xlabel("Sales")
plt.ylabel("Frequency")
plt.show()

# Sales vs Features
numeric_cols = df.select_dtypes(include=['int64','float64']).columns

plt.figure(figsize=(10,6))
sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# ------------ TRAIN A REGRESSION MODEL ------------
# Drop non-numeric columns for regression
df_clean = df.dropna()
X = df_clean.drop("Sales", axis=1).select_dtypes(include=['int64','float64'])
y = df_clean["Sales"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

# ------------ MODEL EVALUATION ------------
y_pred = model.predict(X_test)

print("\n--- MODEL PERFORMANCE ---")
print(f"R² Score: {r2_score(y_test, y_pred):.4f}")
print(f"MAE: {mean_absolute_error(y_test, y_pred):.4f}")
print(f"MSE: {mean_squared_error(y_test, y_pred):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")

# ------------ VISUALIZATION: ACTUAL vs PREDICTED ------------
plt.figure(figsize=(8,5))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales")
plt.show()

# ------------ OPTIONAL TIME SERIES PART (if Date exists) ------------
if "Date" in df.columns:
    df_ts = df.copy()
    df_ts["Date"] = pd.to_datetime(df_ts["Date"])
    df_ts = df_ts.sort_values("Date")

    plt.figure(figsize=(10,5))
    plt.plot(df_ts["Date"], df_ts["Sales"])
    plt.title("Sales Over Time")
    plt.xlabel("Date")
    plt.ylabel("Sales")
    plt.show()

print("\nPipeline Completed Successfully!")
