# Fraud Detection using Machine Learning

A **machine learning project** that analyzes over **6.3 million financial transactions** to detect fraudulent activity.  
The dataset is highly imbalanced (fraud cases < 0.2%), making this a challenging classification problem.  

---

## Project Summary

- **Dataset:** 6.3M financial transactions  
- **Fraud Distribution:** Fraud found only in `TRANSFER` and `CASH_OUT` transaction types  
- **Challenge:** Highly imbalanced dataset (fraud < 0.2%)  

### Steps Performed
1. **Data Cleaning & Feature Engineering**
   - Extracted features from timestamps (day, hour, week).
   - Scaled numerical features.
   - One-Hot Encoding for categorical variables.

2. **Model Training**
   - Train-Test split with stratification to preserve fraud ratio.
   - Tested multiple algorithms:
     - Logistic Regression
     - Random Forest
     - Support Vector Classifier (SVC)
     - K-Nearest Neighbors (KNN)
     - XGBoost
     - Voting Classifier (Ensemble of LR, RF, XGB)

3. **Model Results (approx values)**
   - Logistic Regression → ~99% accuracy (baseline, weak recall for fraud cases).  
   - Random Forest → ~99% accuracy (better fraud detection, handles imbalance).  
   - XGBoost → ~99% accuracy (captures non-linear patterns, strong recall).  
   - KNN & SVC → weak due to large dataset size and imbalance sensitivity.  
   - **Voting Classifier (LR + RF + XGB)** → best balance between **accuracy, recall, and precision**.  

---

## Interpretation of Results

- **Logistic Regression:** Strong baseline accuracy but poor recall → missed many fraud cases.  
- **Random Forest & XGBoost:** High accuracy and better recall → more effective at catching fraud.  
- **KNN & SVC:** Computationally expensive and less effective on large imbalanced datasets.  
- **Voting Classifier:**  
  - Combined strengths of Logistic Regression, Random Forest, and XGBoost.  
  - Achieved the **most balanced performance** across all metrics.  
  - High accuracy with improved recall (fraud detection) and precision (fewer false alarms).  

👉 **Key Insight:**  
In fraud detection, **recall is critical** — missing fraud is costlier than investigating false positives.  
**XGBoost and Voting Classifier are the best models for deployment**, with the ensemble being the most reliable.  

---

## Improvements & Next Steps

- **Modeling Enhancements**
  - Apply **cross-validation** to avoid overfitting.
  - Tune hyperparameters (e.g., depth, `n_estimators`, `learning_rate`).
  - Use **class weights** or **SMOTE** for handling imbalance.

- **Feature Engineering**
  - Add behavioral features:  
    - Customer transaction history  
    - Transaction frequency  
    - Unusual hours of activity  
    - Sudden large transfers  

- **Production Considerations**
  - Continuously **monitor model performance**.  
  - **Retrain regularly** since fraud patterns evolve.  
  - Focus on **precision & recall**, not just accuracy.  

---

## Tech Stack

- **Language:** Python  
- **Data Handling:** Pandas, NumPy  
- **Visualization:** Matplotlib, Seaborn  
- **Machine Learning Models:**  
  - Scikit-learn (Logistic Regression, SVC, KNN, Random Forest, Voting Classifier)  
  - XGBoost  

---
