# 📊 Customer Churn Prediction App

A full-stack machine learning project that predicts telecom customer churn using advanced ML techniques and serves predictions through an interactive, user-friendly Streamlit app.

---

## 🚀 Features

* **Live churn prediction** with a user-friendly interface
* Input only 6 intuitive customer features
* Shows churn probability and risk level (Low / Medium / High)
* Based on a **stacked ensemble model** (LogReg + XGBoost + Random Forest)
* Includes advanced techniques:

  * SMOTE for class balancing
  * Feature engineering based on SHAP insights
  * Model explainability via SHAP (optional)

---

## 🧠 Machine Learning Stack

* 📌 Model: Stacking (LogReg, XGBoost, RF) + Logistic meta-learner
* 🧪 Preprocessing: Label encoding, feature flags, engineered interaction features
* 🧮 Tuning: GridSearchCV on XGBoost
* ⚖️ Balanced dataset using SMOTE
* 🔍 Explanation via SHAP values (optional)

---

## 🖥️ How to Run Locally

### 1. Clone the repo

```bash
git clone https://github.com/your-username/churn-prediction-app.git
cd churn-prediction-app
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the app

```bash
streamlit run app.py
```

---

## 🗂 Project Structure

```
churn-prediction-app/
├── app.py                     # Streamlit interface
├── model/
│   ├── churn_stack_model.pkl  # Trained stacked model
│   └── feature_columns.json   # Feature schema
├── notebook/
│   └── churn_model_pipeline.ipynb  # EDA, modeling, SHAP
├── requirements.txt           # App dependencies
├── README.md                  # This file
```

---

## 🌐 Live Demo (optional)

👉 Hosted on Streamlit Cloud: \[streamlit.io link once deployed]

---

## 👤 Author

**Talel Taieb**
[\[LinkedIn Profile]](www.linkedin.com/in/talel-taieb-824062330)
[\[Portfolio or Website]](https://github.com/taleltaieb)

---

## 📜 License

This project is open source and available under the [MIT License](LICENSE).
