# 📡 Telco Churn Prediction — XGBoost + SHAP (XAI)

> Predict customer churn with 80.7% accuracy using XGBoost and explain every prediction with SHAP explainability — deployed as an interactive Streamlit dashboard.

![Python](https://img.shields.io/badge/Python-3.10-blue)
![XGBoost](https://img.shields.io/badge/XGBoost-2.x-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red)
![SHAP](https://img.shields.io/badge/SHAP-Explainability-green)

---

## 🎯 Project Overview

| Property | Value |
|----------|-------|
| Dataset | IBM Telco Customer Churn (7,043 customers) |
| Algorithm | XGBoost + SMOTE |
| Accuracy | **80.7%** |
| ROC-AUC | **0.84** |
| Explainability | SHAP (Global + Local) |
| Dashboard | Streamlit |

---

## 🚀 Features

- ✅ **Churn Prediction** — Real-time probability with gauge chart
- ✅ **SHAP Waterfall** — Why is THIS customer predicted to churn?
- ✅ **Global Explainability** — Top features driving churn across all customers
- ✅ **Model Performance** — Confusion matrix + ROC curve
- ✅ **Interactive Sidebar** — Adjust customer profile and see instant predictions

---

## 📁 Project Structure
---

## 🛠️ Tech Stack
---

## ⚙️ Run Locally

```bash
# Clone the repo
git clone https://github.com/Umair-khitab/churn-xai.git
cd churn-xai

# Create virtual environment
python -m venv venv
source venv/Scripts/activate   # Windows Git Bash

# Install dependencies
pip install -r requirements.txt

# Launch dashboard
streamlit run app.py
```

---

## 💡 Key Insights

- **Tenure** — Longer customers churn far less
- **Monthly Charges** — Higher bills strongly predict churn
- **Contract Type** — Month-to-month = 3x higher churn risk
- **Tech Support** — Absence increases churn significantly
- **Fiber Optic** — Linked to higher churn (pricing dissatisfaction)

---

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| Accuracy | 80.7% |
| ROC-AUC | 0.84 |
| Precision (No Churn) | 0.86 |
| Recall (No Churn) | 0.83 |
| F1-Score (weighted) | 0.78 |

---

## 👤 Author

**Umair Khitab**
- GitHub: [@Umair-khitab](https://github.com/Umair-khitab)

---

*Built with ❤️ using Python, XGBoost, SHAP and Streamlit*
