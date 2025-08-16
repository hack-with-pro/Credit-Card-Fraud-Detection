# Credit Card Fraud Detection Project

## 1. Overview

This project is a classic machine learning task to build a model that can identify fraudulent credit card transactions. The primary challenge is dealing with a highly imbalanced dataset, where fraudulent transactions are extremely rare (less than 0.2% of the data).

---

## 2. Dataset

The dataset used is the "Credit Card Fraud Detection" dataset from Kaggle. It contains anonymized transaction data from European cardholders over a two-day period.

- **Source:** [Kaggle Dataset Link](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Features:** The dataset includes 28 anonymized features (`V1` to `V28`), plus `Time` and `Amount`.
- **Target:** The `Class` column, where `1` indicates a fraudulent transaction and `0` indicates a legitimate one.

---

## 3. Methodology

The project followed these key steps:
1.  **Data Exploration:** Loaded the data using `pandas` and analyzed the severe class imbalance.
2.  **Data Preparation:** Split the data into an 80% training set and a 20% testing set using a stratified split to preserve the class distribution.
3.  **Model Training:** Trained a `LogisticRegression` model on the training data.
4.  **Model Evaluation:** Evaluated the model's performance on the unseen test data using a Confusion Matrix and Classification Report.

---

## 4. Results

The model's performance was evaluated using a Confusion Matrix. The key goal was to maximize **recall** for the fraudulent class (catching as many frauds as possible) while keeping the number of false alarms low.

**Confusion Matrix Results:**
- **Frauds Caught (True Positives):** 66
- **Frauds Missed (False Negatives):** 32
- **Recall for Fraud:** 67%

This result shows a strong baseline performance, successfully identifying two-thirds of all fraudulent transactions in the test set.

---

## 5. How to Run

1.  Ensure you have Python and the Anaconda distribution installed.
2.  Install the required libraries:
    ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn
    ```
3.  Download the `creditcard.csv` file from the Kaggle link above and place it in the same directory as the notebook.
4.  Run the cells in the `Fraud_Detection.ipynb` notebook file.
