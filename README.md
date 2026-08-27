# Credit Wise – Loan Approval Prediction

A machine learning project that predicts whether a loan application is likely to be approved based on applicant financial, demographic, employment, and loan-related information.

## 📌 Project Overview

Loan approval is an important decision for financial institutions because it involves assessing an applicant's financial profile and ability to repay a loan.

**Credit Wise** explores how machine learning classification algorithms can be used to predict loan approval outcomes from historical application data.

The project covers the complete machine learning workflow:

- Data loading and inspection
- Data cleaning
- Missing-value handling
- Exploratory Data Analysis (EDA)
- Categorical feature encoding
- Feature scaling
- Train-test splitting
- Machine learning model training
- Model evaluation
- Feature engineering
- Model comparison

---

## 📊 Dataset

The dataset contains **1,000 loan applications** and **20 features** describing applicant and loan characteristics.

### Key Features

| Feature | Description |
|---|---|
| `Applicant_Income` | Applicant's income |
| `Coapplicant_Income` | Co-applicant's income |
| `Employment_Status` | Employment status |
| `Age` | Applicant's age |
| `Marital_Status` | Marital status |
| `Dependents` | Number of dependents |
| `Credit_Score` | Applicant's credit score |
| `Existing_Loans` | Number of existing loans |
| `DTI_Ratio` | Debt-to-income ratio |
| `Savings` | Applicant's savings |
| `Collateral_Value` | Value of collateral |
| `Loan_Amount` | Requested loan amount |
| `Loan_Term` | Loan repayment term |
| `Loan_Purpose` | Purpose of the loan |
| `Property_Area` | Property location |
| `Education_Level` | Education level |
| `Gender` | Applicant gender |
| `Employer_Category` | Employer category |
| `Loan_Approved` | Target variable |

### Target Variable

`Loan_Approved`

- `1` → Loan approved
- `0` → Loan not approved

The dataset contains an imbalanced target distribution, with more applications belonging to the non-approved class.

---

## 🔍 Exploratory Data Analysis

The project performs exploratory analysis to understand:

- Distribution of loan approval outcomes
- Numerical feature distributions
- Relationships between variables
- Correlations between numerical features
- Potential patterns in applicant and loan characteristics

A correlation heatmap is also used to visualize relationships between numerical variables.

---

## 🧹 Data Preprocessing

### Missing Values

Missing values are handled separately based on data type.

**Numerical features**

Missing values are replaced using the mean of the respective feature.

**Categorical features**

Missing values are replaced using the most frequently occurring category.

### Categorical Encoding

Categorical variables are converted into numerical representations using **One-Hot Encoding**.

The following categorical features are encoded:

- Employment Status
- Marital Status
- Loan Purpose
- Property Area
- Gender
- Employer Category

### Feature Scaling

Numerical features are standardized using `StandardScaler`.

This ensures that features with different numerical ranges do not disproportionately influence algorithms that are sensitive to feature scale.

---

## 🤖 Machine Learning Models

Three classification algorithms are evaluated:

### 1. Logistic Regression

Logistic Regression is used as a baseline classification model for predicting the probability of loan approval.

### 2. K-Nearest Neighbors (KNN)

KNN predicts the class of an applicant based on the classes of its nearest data points.

The project uses:

```text
n_neighbors = 5
```

### 3. Gaussian Naive Bayes

Gaussian Naive Bayes uses probabilistic relationships between features to classify loan applications.

---

## ⚙️ Feature Engineering

Additional features are created to provide the models with potentially useful nonlinear representations.

### Squared DTI Ratio

```python
DTI_Ratio_sq = DTI_Ratio ** 2
```

### Squared Credit Score

```python
Credit_Score_sq = Credit_Score ** 2
```

These engineered features are then used to retrain and evaluate the models.

---

## 📈 Model Performance

### Initial Model Results

| Model                | Accuracy |  Precision | Recall | F1 Score |
| -------------------- | -------: | ---------: | -----: | -------: |
| Logistic Regression  |    86.5% |     78.33% | 77.05% |   77.69% |
| KNN                  |    76.0% |     62.75% | 52.46% |   57.14% |
| Gaussian Naive Bayes |    86.5% | **80.36%** | 73.77% |   76.92% |

### Results After Feature Engineering

| Model                |  Accuracy |  Precision |     Recall |   F1 Score |
| -------------------- | --------: | ---------: | ---------: | ---------: |
| Logistic Regression  | **88.0%** |     78.46% | **83.61%** | **80.95%** |
| KNN                  |     78.5% |     67.31% |     57.38% |     61.95% |
| Gaussian Naive Bayes |     86.0% | **81.13%** |     70.49% |     75.44% |

### Key Result

After feature engineering:

* **Logistic Regression achieved the highest accuracy: 88.0%**
* **Logistic Regression achieved the highest recall: 83.61%**
* **Logistic Regression achieved the highest F1 score: 80.95%**
* **Gaussian Naive Bayes achieved the highest precision: 81.13%**

This demonstrates that model performance depends on the evaluation metric being prioritized.

---

## 🧮 Evaluation Metrics

The models are evaluated using:

### Accuracy

Percentage of total predictions that are correct.

### Precision

Measures how many predicted approvals were actually correct.

### Recall

Measures how many actual approvals were correctly identified.

### F1 Score

The harmonic mean of precision and recall.

### Confusion Matrix

Used to understand the different types of correct and incorrect predictions.

---

## 🛠️ Technologies Used

* **Python**
* **Pandas**
* **NumPy**
* **Matplotlib**
* **Seaborn**
* **Scikit-learn**
* **Jupyter Notebook**

---

## 📁 Project Structure

```text
credit-wise/
│
├── credit_wise.ipynb
├── loan_approval_data.csv
└── README.md
```

### Files

**`credit_wise.ipynb`**

Contains the complete data analysis, preprocessing, model training, evaluation, and feature engineering workflow.

**`loan_approval_data.csv`**

Contains the loan application dataset used in the project.

**`README.md`**

Project documentation and overview.

---

## 🚀 How to Run

### 1. Clone the repository

```bash
git clone https://github.com/himanshudabas2004-sudo/credit-wise.git
```

### 2. Navigate to the project

```bash
cd credit-wise
```

### 3. Install dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

### 4. Launch Jupyter Notebook

```bash
jupyter notebook
```

### 5. Open

```text
credit_wise.ipynb
```

Make sure `loan_approval_data.csv` is located in the same directory as the notebook.

---

## 💡 Future Improvements

Potential improvements to the project include:

* Using `Pipeline` and `ColumnTransformer` for a more robust preprocessing workflow
* Using stratified train-test splitting
* Hyperparameter tuning
* Cross-validation
* ROC-AUC and PR-AUC evaluation
* Feature importance and model interpretability
* Comparing additional models such as Random Forest, XGBoost, and Gradient Boosting
* Handling class imbalance using appropriate techniques
* Deploying the final model as a web application or API

---

## 👨‍💻 Author

**Himanshu Dabas**

GitHub:
[https://github.com/himanshudabas2004-sudo](https://github.com/himanshudabas2004-sudo)

---

## 📄 License

This project is intended for educational and portfolio purposes.
