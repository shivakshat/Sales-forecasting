import streamlit as st
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np

# -------------------------------
# STREAMLIT PAGE SETTINGS
# -------------------------------
st.set_page_config(page_title="Sales Forecasting Dashboard",
                   layout="wide")

st.title("📈 Sales Forecasting Dashboard")
st.write("A complete dashboard for exploring data, performing SQL queries, training models, and predicting sales.")

# -------------------------------
# LOAD DATA
# -------------------------------
@st.cache_data
def load_data(path):
    return pd.read_csv(path)

file_path = "dataset.csv"   # change if needed
df = load_data(file_path)

st.sidebar.header("Dataset")
st.sidebar.success("Data Loaded Successfully")

# -------------------------------
# SQL DATABASE SETUP
# -------------------------------
conn = sqlite3.connect("sales_db.sqlite")
df.to_sql("sales_table", conn, if_exists="replace", index=False)

# -------------------------------
# SIDEBAR OPTIONS
# -------------------------------
st.sidebar.header("Navigation")
section = st.sidebar.radio("Go to:", 
                           ["Dataset", "EDA", "SQL Query", "Model Training", "Prediction"])

# ============================================================
# SECTION 1 - DATASET
# ============================================================
if section == "Dataset":
    st.subheader("📄 Dataset Preview")
    st.dataframe(df)

    st.write("### Summary Statistics")
    st.write(df.describe())

# ============================================================
# SECTION 2 - EDA
# ============================================================
elif section == "EDA":
    st.subheader("🔎 Exploratory Data Analysis")

    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns

    st.write("### Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(8,6))
    sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    st.write("### Sales Distribution")
    fig2, ax2 = plt.subplots(figsize=(7,4))
    sns.histplot(df["Sales"], kde=True, ax=ax2)
    st.pyplot(fig2)

    if "Date" in df.columns:
        st.write("### Sales Over Time")
        df_ts = df.copy()
        df_ts["Date"] = pd.to_datetime(df_ts["Date"])
        df_ts = df_ts.sort_values("Date")

        fig3, ax3 = plt.subplots(figsize=(10,4))
        ax3.plot(df_ts["Date"], df_ts["Sales"])
        ax3.set_title("Sales Trend")
        st.pyplot(fig3)

# ============================================================
# SECTION 3 - SQL QUERY
# ============================================================
elif section == "SQL Query":
    st.subheader("🗄 SQL Query Runner")
    st.write("Write your SQL query below:")

    user_query = st.text_area("SQL Query", 
                              "SELECT * FROM sales_table LIMIT 5;")

    if st.button("Run Query"):
        try:
            result = pd.read_sql_query(user_query, conn)
            st.success("Query Executed Successfully!")
            st.dataframe(result)
        except Exception as e:
            st.error(f"Error: {e}")

# ============================================================
# SECTION 4 - MODEL TRAINING
# ============================================================
elif section == "Model Training":
    st.subheader("🤖 Model Training")

    df_clean = df.dropna()
    X = df_clean.drop("Sales", axis=1).select_dtypes(include=['int64','float64'])
    y = df_clean["Sales"]

    test_size = st.slider("Test Size (in %)", 10, 50, 20)
    random_state = st.number_input("Random State", 0, 1000, 42)

    if st.button("Train Model"):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size/100, random_state=random_state
        )

        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        st.write("### Model Performance")
        st.write(f"**R² Score:** {r2_score(y_test, y_pred):.4f}")
        st.write(f"**MAE:** {mean_absolute_error(y_test, y_pred):.4f}")
        st.write(f"**MSE:** {mean_squared_error(y_test, y_pred):.4f}")
        st.write(f"**RMSE:** {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")

        # Visualization
        st.write("### Actual vs Predicted")
        fig4, ax4 = plt.subplots(figsize=(6,4))
        ax4.scatter(y_test, y_pred)
        ax4.set_xlabel("Actual")
        ax4.set_ylabel("Predicted")
        ax4.set_title("Actual vs Predicted")
        st.pyplot(fig4)

        st.session_state["model"] = model
        st.session_state["features"] = list(X.columns)

# ============================================================
# SECTION 5 - PREDICTION
# ============================================================
elif section == "Prediction":
    st.subheader("📌 Predict Sales")

    if "model" not in st.session_state:
        st.warning("⚠ Please train the model first in the 'Model Training' section.")
    else:
        model = st.session_state["model"]
        features = st.session_state["features"]

        st.write("### Enter Feature Values:")
        input_data = {}

        for col in features:
            input_data[col] = st.number_input(f"{col}:", value=0.0)

        if st.button("Predict"):
            user_df = pd.DataFrame([input_data])
            prediction = model.predict(user_df)[0]
            st.success(f"Predicted Sales: **{prediction:.2f}**")
