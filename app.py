import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path

# =========================================
# PAGE CONFIG
# =========================================

st.set_page_config(
    page_title="AdaBoost Classifier App",
    layout="wide"
)

# =========================================
# PATHS
# =========================================

BASE_DIR = Path(__file__).resolve().parent

model_path = BASE_DIR / "models" / "adaboost_classifier.pkl"
scaler_path = BASE_DIR / "models" / "scaler.pkl"
data_path = BASE_DIR / "data/raw/data.csv"

# =========================================
# LOAD DATASET
# =========================================

df = pd.read_csv(data_path)

# =========================================
# LOAD MODEL
# =========================================

try:
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

except Exception as e:

    st.error(f"Error Loading Model: {e}")

    model = None
    scaler = None

# =========================================
# SIDEBAR
# =========================================

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Go To",
    [
        "Dataset Overview",
        "EDA",
        "Prediction"
    ]
)

# =========================================
# DATASET OVERVIEW
# =========================================

if page == "Dataset Overview":

    st.title("📊 Dataset Overview")

    st.subheader("Dataset Shape")
    st.write(df.shape)

    st.subheader("First 5 Rows")
    st.dataframe(df.head())

    st.subheader("Column Names")
    st.write(df.columns.tolist())

    st.subheader("Statistical Summary")
    st.dataframe(df.describe())

    st.subheader("Missing Values")
    st.write(df.isnull().sum())

# =========================================
# EDA PAGE
# =========================================

elif page == "EDA":

    st.title("📈 Exploratory Data Analysis")

    # =====================================
    # TARGET DISTRIBUTION
    # =====================================

    st.subheader("Target Variable Distribution")

    fig1, ax1 = plt.subplots(figsize=(6,4))

    sns.countplot(
        x='target',
        data=df,
        ax=ax1
    )

    st.pyplot(fig1)

    # =====================================
    # FEATURE DISTRIBUTIONS
    # =====================================

    st.subheader("Feature Distributions")

    fig2 = df.hist(
        figsize=(20,18),
        bins=30
    )

    st.pyplot(plt.gcf())

    # =====================================
    # CORRELATION HEATMAP
    # =====================================

    st.subheader("Correlation Heatmap")

    fig3, ax3 = plt.subplots(figsize=(18,12))

    sns.heatmap(
        df.corr(),
        cmap='coolwarm',
        annot=False,
        ax=ax3
    )

    st.pyplot(fig3)

    # =====================================
    # BOXPLOT
    # =====================================

    st.subheader("Outlier Detection")

    selected_column = st.selectbox(
        "Select Feature",
        df.columns[:-1]
    )

    fig4, ax4 = plt.subplots(figsize=(10,4))

    sns.boxplot(
        x=df[selected_column],
        ax=ax4
    )

    st.pyplot(fig4)

# =========================================
# PREDICTION PAGE
# =========================================

elif page == "Prediction":

    st.title("🤖 AdaBoost Classification Prediction")

    if model is None or scaler is None:

        st.error("Model or Scaler file not found!")

    else:

        st.subheader("Enter Feature Values")

        input_data = []

        # Dynamic Input Fields
        for col in df.columns[:-1]:

            value = st.number_input(
                f"{col}",
                value=0.0,
                format="%.4f"
            )

            input_data.append(value)

        # =================================
        # PREDICTION BUTTON
        # =================================

        if st.button("Predict"):

            try:

                # Convert to DataFrame
                input_df = pd.DataFrame(
                    [input_data],
                    columns=df.columns[:-1]
                )

                # Scale Input
                input_scaled = scaler.transform(input_df)

                # Prediction
                prediction = model.predict(input_scaled)

                # Probability
                probability = model.predict_proba(input_scaled)

                # =========================
                # DISPLAY RESULTS
                # =========================

                st.subheader("Prediction Result")

                if prediction[0] == 1:

                    st.success("Positive Class Detected")

                else:

                    st.error("Negative Class Detected")

                # =========================
                # PROBABILITY DISPLAY
                # =========================

                st.subheader("Prediction Probability")

                prob_class_0 = probability[0][0]
                prob_class_1 = probability[0][1]

                st.write(
                    f"Class 0 Probability: {prob_class_0:.4f}"
                )

                st.write(
                    f"Class 1 Probability: {prob_class_1:.4f}"
                )

                st.progress(float(prob_class_1))

            except Exception as e:

                st.error(f"Prediction Error: {e}")

# =========================================
# FOOTER
# =========================================

st.sidebar.markdown("---")
st.sidebar.write("AdaBoost Classifier Streamlit App")