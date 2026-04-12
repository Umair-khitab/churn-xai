import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import os

st.set_page_config(page_title="Churn Prediction · XAI", page_icon="📡", layout="wide")

@st.cache_resource
def load_artifacts():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("explainer.pkl", "rb") as f:
        explainer = pickle.load(f)
    with open("features.pkl", "rb") as f:
        features = pickle.load(f)
    return model, explainer, features

model, explainer, feature_names = load_artifacts()

st.sidebar.header("🧑‍💼 Customer Profile")
tenure          = st.sidebar.slider("Tenure (months)", 0, 72, 12)
monthly_charges = st.sidebar.slider("Monthly Charges ($)", 18, 120, 65)
total_charges   = monthly_charges * tenure
contract        = st.sidebar.selectbox("Contract", ["Month-to-month","One year","Two year"])
internet        = st.sidebar.selectbox("Internet Service", ["DSL","Fiber optic","No"])
tech_support    = st.sidebar.selectbox("Tech Support", ["Yes","No","No internet service"])
online_sec      = st.sidebar.selectbox("Online Security", ["Yes","No","No internet service"])
paperless       = st.sidebar.checkbox("Paperless Billing", True)
senior          = st.sidebar.checkbox("Senior Citizen", False)

def build_input():
    row = dict.fromkeys(feature_names, 0)
    row['tenure']           = tenure
    row['MonthlyCharges']   = monthly_charges
    row['TotalCharges']     = total_charges
    row['SeniorCitizen']    = int(senior)
    row['PaperlessBilling'] = int(paperless)
    if contract == "One year":
        if 'Contract_One year' in row: row['Contract_One year'] = 1
    elif contract == "Two year":
        if 'Contract_Two year' in row: row['Contract_Two year'] = 1
    if internet == "Fiber optic":
        if 'InternetService_Fiber optic' in row: row['InternetService_Fiber optic'] = 1
    elif internet == "No":
        if 'InternetService_No' in row: row['InternetService_No'] = 1
    if tech_support == "No internet service":
        if 'TechSupport_No internet service' in row: row['TechSupport_No internet service'] = 1
    elif tech_support == "Yes":
        if 'TechSupport_Yes' in row: row['TechSupport_Yes'] = 1
    if online_sec == "No internet service":
        if 'OnlineSecurity_No internet service' in row: row['OnlineSecurity_No internet service'] = 1
    elif online_sec == "Yes":
        if 'OnlineSecurity_Yes' in row: row['OnlineSecurity_Yes'] = 1
    return pd.DataFrame([row])

input_df = build_input()

st.title("📡 Telco Churn Prediction — XGBoost + SHAP")
st.caption("XAI-powered customer retention intelligence")

tab1, tab2, tab3 = st.tabs(["🎯 Prediction", "🌐 Global Explainability", "📊 Model Performance"])

with tab1:
    proba = model.predict_proba(input_df)[0][1]
    pred  = int(proba >= 0.5)
    risk  = "🔴 HIGH RISK" if proba > 0.7 else ("🟡 MEDIUM RISK" if proba > 0.4 else "🟢 LOW RISK")
    c1, c2, c3 = st.columns(3)
    c1.metric("Churn Probability", f"{proba:.1%}")
    c2.metric("Prediction", "Will Churn" if pred else "Stays")
    c3.metric("Risk Level", risk)
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=round(proba*100,1),
        title={'text': "Churn Risk %"},
        gauge={'axis': {'range': [0,100]},
               'bar': {'color': "crimson" if proba > 0.5 else "steelblue"},
               'steps': [{'range':[0,40],'color':'#d4efdf'},
                          {'range':[40,70],'color':'#fdebd0'},
                          {'range':[70,100],'color':'#fadbd8'}],
               'threshold': {'line':{'color':'red','width':4},'thickness':0.75,'value':50}}))
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)
    st.subheader("🔍 Why this prediction? (SHAP Waterfall)")
    shap_vals = explainer.shap_values(input_df)
    fig2, ax = plt.subplots(figsize=(10,5))
    shap.waterfall_plot(shap.Explanation(
        values=shap_vals[0], base_values=explainer.expected_value,
        data=input_df.iloc[0], feature_names=feature_names), show=False)
    st.pyplot(fig2)

with tab2:
    if os.path.exists("shap_bar.png"):
        st.subheader("📊 Feature Importance (mean |SHAP|)")
        st.image("shap_bar.png", use_container_width=True)
    if os.path.exists("shap_summary.png"):
        st.subheader("🐝 SHAP Beeswarm")
        st.image("shap_summary.png", use_container_width=True)

with tab3:
    c1, c2 = st.columns(2)
    with c1:
        if os.path.exists("confusion_matrix_roc.png"):
            st.image("confusion_matrix_roc.png", use_container_width=True)
    with c2:
         st.metric("Accuracy",  "80.7%")
         st.metric("ROC-AUC",   "0.84")
         st.metric("Precision", "0.86")
         st.metric("Recall",    "0.83")
