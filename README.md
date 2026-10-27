# 🔍 Interpretable Machine Learning with SHAP & LIME

### Opening the black box of a customer churn model

![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit_learn-F7931E?logo=scikitlearn&logoColor=white)
![SHAP](https://img.shields.io/badge/SHAP-Explainability-E4405F)
![LIME](https://img.shields.io/badge/LIME-Explainability-1B9AAA)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?logo=jupyter&logoColor=white)

🇮🇩 [Baca dalam Bahasa Indonesia](README_ID.md)

---

## 📌 Project Overview

This project focuses on **Interpretable Machine Learning (IML)** using the **Telco Customer Churn** dataset (7,043 customers).

The main goal is not only to predict which customers are likely to churn, but also to understand **why the model makes a particular prediction**, and to turn those reasons into actions a retention team can use.

The project explores several approaches to machine learning interpretability, including:

* Glass Box vs Black Box models
* Logistic Regression coefficients
* Gradient Boosting
* Random Forest feature importance
* Permutation Importance
* Partial Dependence Plot (PDP)
* Individual Conditional Expectation (ICE)
* LIME
* SHAP
* SHAP Force Plot
* SHAP Waterfall Plot
* SHAP Summary / Beeswarm Plot
* SHAP Dependence Plot
* Business oriented model explanations

The project also compares **LIME and SHAP** in terms of explanation consistency, computational speed, and local vs global interpretability.

---

## ⚡ At a Glance

**🎯 Goal.** Understand why a model predicts churn, and turn those reasons into business actions.

**🛠️ How.** Compare a glass box and a black box model, then dissect a Random Forest with permutation importance, PDP, LIME, and SHAP.

**🏆 Result.**
* Tenure is the strongest churn driver, followed by fiber optic service and contract type.
* Month to month contracts churn at **42.7%**, versus only **2.8%** for two year contracts.
* A simple, explainable model was almost as accurate as a complex one (AUC **0.846** vs **0.847**).
* Recommendation: a 90 day onboarding program and incentives toward longer contracts, validated with an A/B test.

---

## 🎯 Objectives

* Compare a **glass box** model with a **black box** model and discuss the trade off between interpretability and accuracy
* Identify the **global drivers** of churn with feature importance and PDP
* Explain **individual predictions** with LIME and SHAP, and compare the two methods
* Find **overall patterns** with SHAP summary and dependence plots
* Translate technical findings into **business insight** for non technical stakeholders

---

## 🗂️ Dataset

* **Source:** Telco Customer Churn (IBM sample data, available on Kaggle)
* **Size:** 7,043 customers, 21 columns
* **Target:** `Churn` (about 26.5% of customers churned)
* **Cleaning:**
  * `TotalCharges` converted to numbers (11 blank values belong to new customers with zero tenure and were set to 0)
  * `customerID` dropped
  * "No internet service" and "No phone service" merged into "No"
  * One hot encoding, resulting in 23 features
* **Split:** 75% train and 25% test, stratified by the target

---

## 🧪 Methodology

1. **Glass box vs black box:** Logistic Regression compared with Gradient Boosting
2. **Global explanations:** Random Forest with permutation importance and PDP/ICE plots
3. **Local explanations:** LIME and SHAP (force and waterfall plots) for one high risk customer
4. **Global SHAP view:** summary (beeswarm) and dependence plots
5. **Business communication:** executive summary and recommendations

---

## 📊 Key Results

### 1️⃣ Glass Box vs Black Box

* **Logistic Regression:** AUC 0.846, accuracy 0.807, recall (churn) 0.557, precision (churn) 0.663
* **Gradient Boosting:** AUC 0.847, accuracy 0.798, recall (churn) 0.497, precision (churn) 0.659

On this dataset the black box brings no meaningful accuracy gain, so the explainable model is the more sensible choice, especially for regulated decisions such as credit.

![Logistic Regression coefficients](images/01_logreg_coefficients.png)

### 2️⃣ What Drives Churn

Random Forest (AUC 0.848, accuracy 0.801). Top drivers by permutation importance:

* `tenure` (AUC drops by about 0.048 when shuffled)
* `InternetService_Fiber optic` (about 0.022)
* `Contract_Two year` (about 0.019)

![Feature importance](images/02_feature_importance.png)

PDP shows the risk is highest in the first year: about **41%** churn probability in month one, falling to about 30% by month 11 and settling around **20%** after two years.

![PDP and ICE](images/03_pdp_ice.png)

### 3️⃣ Explaining One Customer: LIME vs SHAP

For a customer with a **79%** churn probability (the average is 26%), both methods point to a very short tenure, fiber optic service, and a month to month contract.

![LIME explanation](images/04_lime_explanation.png)
![SHAP force plot](images/05_shap_force_plot.png)
![SHAP waterfall](images/06_shap_waterfall.png)

### 4️⃣ Patterns Across All Customers

Long tenure and long contracts lower churn risk, while fiber optic service, electronic check payment, and high monthly charges raise it.

![SHAP beeswarm](images/07_shap_beeswarm.png)
![SHAP dependence plot for tenure](images/08_shap_dependence_tenure.png)

---

## 🆚 LIME vs SHAP

* **Consistency:** 3 of the top 5 drivers match, with the same direction. SHAP values add up exactly to the model prediction, while LIME is an approximation that depends on random sampling (its top 3 reasons stayed the same across 3 seeds).
* **Speed:** SHAP (TreeExplainer) took about 0.08 seconds per customer and 37 seconds for the full test set. LIME took about 0.44 seconds per customer, an estimated 13 minutes for the same set.
* **Scope:** SHAP supports local and global explanations. LIME is local only.
* **When to use which:** SHAP for tree based models and auditable results. LIME when no fast SHAP explainer exists for the model.

---

## 💡 Business Recommendations

* 🚀 Run a **90 day program** for new customers (onboarding, connection quality check, follow up). Churn is 47% for tenure of 12 months or less, versus 17% for longer tenures.
* 📝 Offer discounts or upgrades to move high risk monthly customers to **1 or 2 year contracts**.
* 🔧 Review **pricing and service quality** for fiber optic customers.
* 🧪 Validate with an **A/B test**. The model shows patterns, not causes.

---

## ⚠️ Limitations

* SHAP and LIME explain the model's behavior, not real world causality.
* Results come from a single train and test split of one public dataset.
* Timing figures depend on hardware.

---

## 📁 Repository Structure

```
telco_churn_lime_shap/
├── README.md
├── README_ID.md
├── requirements.txt
├── churn_lime_shap.ipynb
├── WA_Fn-UseC_-Telco-Customer-Churn.csv
└── images/
```

---

## 🚀 How to Run

1. Clone this repository and install the dependencies:
   ```
   pip install -r requirements.txt
   ```
2. Make sure the CSV file is available and, if needed, update the file path in the data loading cell of the notebook.
3. Open `churn_lime_shap.ipynb` and run the cells from top to bottom.

---

## 🛠️ Tech Stack

Python, pandas, NumPy, scikit-learn, SHAP, LIME, Matplotlib, Jupyter

---

## 👤 Author

**I Made Bimantoro Mastra**, data science learner with a background in 3D art and animation.
Developed as a learning project during a data science bootcamp.
