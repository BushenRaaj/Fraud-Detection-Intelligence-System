# -*- coding: utf-8 -*-
"""
Created on Sat Dec 27 18:23:46 2025

@author: bhush
"""

import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Fraud Detection Intelligence",
    page_icon="💳",
    layout="wide"
)

# =========================
# PATHS
# =========================

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(BASE_DIR, "..", "data", "Fraud_Analysis_Dataset.csv")
RF_MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "fraud_rf_model.joblib")
GB_MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "fraud_gb_model.joblib")
SCALER_PATH = os.path.join(BASE_DIR, "..", "models", "scaler.joblib")
TYPE_ENCODER_PATH = os.path.join(BASE_DIR, "..", "models", "type_encoder.joblib")
TEST_DATA_PATH = os.path.join(BASE_DIR, "..", "data", "model_test_data.csv")


#DATA_PATH = r"D:\Fraud-Detection-Intelligence-System\data\Fraud_Analysis_Dataset.csv"
#RF_MODEL_PATH = r"D:/Fraud-Detection-Intelligence-System/models/fraud_rf_model.joblib"
#GB_MODEL_PATH = r"D:\Fraud-Detection-Intelligence-System\models\fraud_gb_model.joblib"
#SCALER_PATH = r"D:\Fraud-Detection-Intelligence-System\models\scaler.joblib"
#TYPE_ENCODER_PATH = r"D:\Fraud-Detection-Intelligence-System\models\type_encoder.joblib"
#TEST_DATA_PATH = r"D:\Fraud-Detection-Intelligence-System\data\model_test_data.xlsx"

# =========================
# LOAD DATA & MODELS
# =========================
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.lower().str.strip()
    return df

@st.cache_resource
def load_models():
    rf = joblib.load(RF_MODEL_PATH)
    gb = joblib.load(GB_MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    encoder = joblib.load(TYPE_ENCODER_PATH)
    return rf, gb, scaler, encoder

df = load_data()
rf_model, gb_model, scaler, type_encoder = load_models()

fraud_col = "isfraud"

# =========================
# HEADER
# =========================
st.markdown("""
<div style="background:linear-gradient(90deg,#020617,#0f172a);
            padding:25px;border-radius:15px">
<h1 style="color:white;text-align:center;">💳 Fraud Detection Intelligence System</h1>
<p style="color:#cbd5e1;text-align:center;font-size:17px">
Real-Time Transaction Risk Assessment Dashboard
</p>
</div>
""", unsafe_allow_html=True)

st.divider()

# =========================
# KPI SUMMARY
# =========================
k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Transactions", f"{len(df):,}")
k2.metric("Fraud Transactions", f"{df[fraud_col].sum():,}")
k3.metric("Fraud Rate", f"{df[fraud_col].mean()*100:.2f}%")
k4.metric("Models", "RF + GB")


# =========================
# SIDEBAR – TRANSACTION INPUT GENERATOR
# =========================
#st.sidebar.title("🧾 Transaction Input Generator")

st.sidebar.title("🎛️ Transaction Simulator")
st.sidebar.caption("Simulate a transaction to assess fraud risk")

#model_choice = st.sidebar.radio(
 #   "🧠 Select Model",
  #  ["Random Forest", "Gradient Boosting"]
#)

st.sidebar.header("⚙️ Model Selection")
model_choice = st.sidebar.radio(
    "Select Model",
   # ("Random Forest (Primary)", "Gradient Boosting (Challenger)")
   ("Random Forest (High Recall)", "Gradient Boosting (High Precision)")
)

st.sidebar.header("🧾 Transaction Details")

step = st.sidebar.number_input("⏱️ Transaction Step", min_value=0, value=500)
amount = st.sidebar.number_input("💰 Transaction Amount", min_value=1.0, value=5000.0)

st.sidebar.header("🧾 Sender Details")

oldbalanceorg = st.sidebar.number_input("🏦 Old Balance", min_value=0.0, value=10000.0)
newbalanceorig = st.sidebar.number_input("🏦 New Balance", min_value=0.0, value=8000.0)

st.sidebar.header("🧾 Receiver Details")

oldbalancedest = st.sidebar.number_input("🏦 Old Balance", min_value=0.0, value=20000.0)
newbalancedest = st.sidebar.number_input("🏦 New Balance", min_value=0.0, value=25000.0)

transaction_types = list(type_encoder.classes_)
selected_type = st.sidebar.selectbox("🔁 Transaction Type", transaction_types)
type_enc = type_encoder.transform([selected_type])[0]

analyze = st.sidebar.button("🔍 Analyze Transaction", use_container_width=True)

# =========================
# INPUT DATA
# =========================
input_df = pd.DataFrame({
    "step": [step],
    "amount": [amount],
    "oldbalanceorg": [oldbalanceorg],
    "newbalanceorig": [newbalanceorig],
    "oldbalancedest": [oldbalancedest],
    "newbalancedest": [newbalancedest],
    "type_enc": [type_enc]
})

input_df = input_df[scaler.feature_names_in_]
input_scaled = scaler.transform(input_df)

# =========================
# MAIN LAYOUT
# =========================
left, right = st.columns([2, 1])

with left:
    st.subheader("🧾 Transaction Overview")
    st.dataframe(input_df, use_container_width=True)

with right:
    st.subheader("📊 Prediction Analysis")

    if analyze:
        model = rf_model if model_choice == "Random Forest" else gb_model

        pred = model.predict(input_scaled)[0]
        prob = model.predict_proba(input_scaled)[0][1]

        risk = "HIGH" if prob > 0.7 else "MEDIUM" if prob > 0.4 else "LOW"

        if pred == 1:
            st.error(f"🚨 Fraud Detected | Probability: {prob:.2%} | Risk: {risk}")
        else:
            st.success(f"✅ Legitimate Transaction | Probability: {prob:.2%} | Risk: {risk}")

        st.progress(int(prob * 100))

# =========================
# FINANCIAL LOSS ESTIMATION
# =========================
if analyze:
    st.divider()
    st.subheader("💰 Financial Impact Estimation")

    AVG_FRAUD_LOSS = 5000
    c1, c2, c3 = st.columns(3)

    c1.metric("Expected Fraud Loss", f"₹{prob * AVG_FRAUD_LOSS:,.0f}")
    c2.metric("Loss Prevented", f"₹{(1 - prob) * AVG_FRAUD_LOSS:,.0f}")
    c3.metric("Recommended Action",
              "Block" if prob > 0.7 else "Monitor" if prob > 0.4 else "Approve")

# =========================
# CONFUSION MATRIX & ROC
# =========================
st.divider()
st.subheader("📈 Model Performance Evaluation")

tab1, tab2, tab3 = st.tabs(
    ["Confusion Matrix", "ROC Curve", "Fraud Distribution"]
)


test_df = pd.read_csv(TEST_DATA_PATH)
test_df.columns = test_df.columns.str.lower().str.strip()

possible_labels = ["isfraud", "is_fraud", "fraud", "class", "target"]
fraud_label = next((c for c in possible_labels if c in test_df.columns), None)

if fraud_label is None:
    st.error("❌ Fraud label column not found in model_test_data.csv")
    st.stop()

X_test = test_df.drop(columns=[fraud_label])
y_test = test_df[fraud_label]

X_test = X_test[scaler.feature_names_in_]
X_test_scaled = scaler.transform(X_test)

y_prob = rf_model.predict_proba(X_test_scaled)[:, 1]
y_pred = (y_prob >= 0.5).astype(int)

with tab1:
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(3, 1))  # 🔹 Increased size
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_title("Confusion Matrix")
    st.pyplot(fig, use_container_width=True)

with tab2:
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(3, 1))  # 🔹 Wider for ROC
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_title("ROC Curve")
    ax.legend()
    st.pyplot(fig, use_container_width=True)

with tab3:
    st.subheader("📊 Fraud Distribution")

    fig, ax = plt.subplots(figsize=(3, 1))  # 🔹 Wider bar chart
    df[fraud_col].value_counts().plot(kind="bar", ax=ax)
    ax.set_xticklabels(["Legitimate", "Fraud"], rotation=0)
    ax.set_ylabel("Count")
    ax.set_title("Fraud vs Legitimate Transactions")
    st.pyplot(fig, use_container_width=True)


# =========================
# FOOTER
# =========================
st.divider()
st.caption("End-to-End Fraud Detection ML Project | Final Capstone Dashboard")
