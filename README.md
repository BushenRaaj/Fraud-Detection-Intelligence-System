# 💳 Fraud Detection Intelligence System  
### End-to-End Machine Learning Solution for Real-Time Mobile Transaction Fraud Detection

## 🌐 Live Demo
👉 https://fraud-detection-intelligence-system-dbcs6mt83ydwsaamhdavbt.streamlit.app/

---

## 📌 Project Summary

This project implements a complete **Fraud Detection Intelligence System** designed to identify fraudulent mobile financial transactions in real time.

The solution covers the full machine learning lifecycle:

✔ Data preprocessing & feature engineering  
✔ Imbalanced data handling (Fraud vs Legitimate)  
✔ Model training & evaluation  
✔ Model comparison & performance optimization  
✔ Real-time fraud risk scoring using Streamlit  
✔ Business Intelligence dashboard using Power BI  
✔ Financial impact estimation  

This system is built with a strong focus on **maximizing recall** to minimize missed fraud cases while maintaining operational efficiency.

---

## 🎯 Business Problem

Mobile financial transactions are increasingly vulnerable to fraud.  
Traditional rule-based systems struggle with:

- Static fraud rules
- Highly imbalanced datasets
- Delayed detection
- Increasingly sophisticated fraud patterns

This project leverages machine learning to detect hidden transaction anomalies and assign risk-based decisions in real time.

---

## 🏗 System Architecture

```
Transaction Data
        ↓
Data Preprocessing & Feature Engineering
        ↓
Machine Learning Models (RF, GB)
        ↓
Real-Time Inference Engine (Streamlit)
        ↓
Risk-Based Decision Layer
        ↓
Business Insights Dashboard (Power BI)
```

---

## 📊 Dataset Overview

- Historical mobile financial transaction dataset
- Key Features:
  - Transaction Amount
  - Transaction Type
  - Step (Time Index)
  - Sender & Receiver Balances
  - Balance Changes
- Target Variable:
  - `isFraud` (0 = Legitimate, 1 = Fraudulent)

The dataset reflects realistic fraud imbalance, requiring strategic model optimization.

---

## 🧠 Machine Learning Models

Models evaluated:

- Logistic Regression
- Random Forest
- Gradient Boosting

### 📈 Evaluation Metrics

- Recall (Primary Focus)
- Precision
- F1-Score
- ROC-AUC
- Confusion Matrix

Tree-based models demonstrated improved recall while controlling false positives.

---

## 🚀 Real-Time Fraud Simulation (Streamlit App)

The Streamlit application enables:

- Transaction input simulation
- Real-time fraud probability scoring
- Risk categorization:
  - High Risk → Block
  - Medium Risk → Monitor
  - Low Risk → Approve
- Financial loss estimation
- Model performance visualization (ROC, Confusion Matrix)

### ▶ Run Locally

```bash
cd app
streamlit run streamlit_app.py
```

---

## 📊 Business Intelligence Dashboard (Power BI)

The Power BI dashboard provides:

- Fraud trend visualization
- Fraud rate analysis
- Financial impact estimation
- Operational KPIs
- Risk distribution insights

---

## 💰 Financial Impact Layer

The system translates predictions into measurable business value:

- Expected fraud loss estimation
- Loss prevention insights
- Risk-based operational recommendations

This bridges the gap between machine learning output and business decision-making.

---

## 🛠 Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Matplotlib
- Seaborn
- Power BI
- Joblib

---

## 📁 Project Structure

```
Fraud-Detection-Intelligence-System/
│
├── data/
├── models/
├── notebooks/
├── app/
├── dashboard/
├── presentation/
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 🔍 Key Highlights

- End-to-End ML Pipeline
- Imbalanced Data Handling
- Model Comparison & Optimization
- Real-Time Deployment Interface
- Business-Oriented Interpretation
- Clean GitHub Structure

---

## 👤 Author

**Bushen Raaj Y**  
Data Science & Artificial Intelligence  

---

## 📌 Future Enhancements

- Cloud deployment (AWS / Streamlit Cloud)
- Model monitoring & drift detection
- Integration with real-time APIs
- Advanced ensemble methods (XGBoost)

---

⭐ If you found this project insightful, feel free to connect or provide feedback.
