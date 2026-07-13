# Applied AI & ML Capstone Project

## End-to-End Data Intelligence System

This project is developed as part of the Applied AI & ML Capstone program.

The goal of this project is to build a complete machine learning pipeline starting from raw dataset acquisition and exploratory analysis to advanced predictive modeling and LLM-powered explanations.

The project covers:

- Data cleaning and exploratory analysis
- Supervised machine learning
- Ensemble learning and model optimization
- ML pipelines and model serialization
- LLM-powered model explanation system

---

# Project Structure


Applied-AI-ML-Capstone/

│
├── Part-1-Data-Acquisition-and-EDA/
│ ├── notebook.ipynb
│ ├── cleaned_data.csv
│ └── README.md
│
├── Part-2-Supervised-Machine-Learning/
│ ├── Part2_Model_Training.ipynb
│ └── README.md
│
├── Part-3-Ensemble-Models-and-Pipelines/
│ ├── Part3_Advanced_Modeling.ipynb
│ ├── best_model.pkl
│ └── README.md
│
├── Part-4-LLM-Powered-Data-Intelligence/
│ ├── Part4_LLM_Explanation.ipynb
│ └── README.md
│
└── README.md


---

# Dataset

The project uses the Ames Housing Dataset.

The dataset contains property-related features used to predict house prices.

The dataset includes:

- Numerical features
- Categorical features
- Housing characteristics
- Sale price information

The target variable used for regression:


SalePrice


A binary classification target was created by applying median binarization:


y_clf = (SalePrice > SalePrice.median()).astype(int)


Where:

- 0 → Lower priced houses
- 1 → Higher priced houses

---

# Part 1 — Data Acquisition and Exploratory Data Analysis

## Objective

The first stage focused on preparing a clean dataset suitable for machine learning.

## Tasks Completed

### Data Loading

- Loaded the raw dataset using pandas.
- Examined:
  - Dataset shape
  - Column types
  - Initial records

### Data Cleaning

Performed:

- Missing value analysis
- Duplicate detection
- Missing value handling
- Data consistency checks

Numerical missing values were handled using median imputation.

Median was selected because it is less affected by extreme values and outliers.

### Exploratory Data Analysis

Performed analysis on:

- Feature distributions
- Correlations
- Numerical relationships
- Categorical feature patterns

The cleaned dataset was saved as:


cleaned_data.csv


---

# Part 2 — Supervised Machine Learning

## Objective

The objective of Part 2 was to build regression and classification models using the cleaned dataset.

---

# Regression Task

## Target

Regression target:


SalePrice


## Models Used

### Linear Regression

Evaluation metrics:

- Mean Squared Error (MSE)
- R² Score

### Ridge Regression

Ridge regression was applied using:


alpha = 1.0


to reduce coefficient instability and control overfitting.

---

## Regression Results

| Model | MSE | R² Score |
|---|---|---|
| Linear Regression | 8.34e8 | 0.895955 |
| Ridge Regression | 8.29e8 | 0.896499 |

Ridge regression slightly improved performance by applying L2 regularization.

---

# Classification Task

## Target Creation

The regression target was converted into a binary classification problem:


SalePrice > median → 1
SalePrice <= median → 0


The classes were balanced:


Class 0: 1467
Class 1: 1463


---

## Model Used

Logistic Regression

Techniques applied:

- Train-test split
- Feature scaling using StandardScaler
- Class imbalance checking
- Probability prediction using predict_proba()

---

## Classification Evaluation

Metrics calculated:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
- ROC Curve
- AUC Score

Decision threshold analysis was also performed between:


0.30 - 0.70


The optimal threshold was evaluated using F1-score.

---

## Regularization Experiment

Logistic Regression was compared using:

- C = 1.0
- C = 0.01

The impact of stronger regularization on performance was analyzed.

---

## Bootstrap Confidence Interval

500 bootstrap samples were generated to estimate the reliability of AUC differences between models.

The confidence interval was calculated using:

- Mean AUC difference
- 2.5 percentile
- 97.5 percentile

---

# Part 3 — Ensemble Models and Advanced Pipeline

## Objective

Part 3 focused on improving model robustness using ensemble techniques, hyperparameter tuning, and reusable ML pipelines.

---

# Decision Tree Models

Two decision tree models were trained.

## Uncontrolled Decision Tree

Results:


Training Accuracy: 1.0
Testing Accuracy: 0.9095


The large train-test gap indicated overfitting.

---

## Controlled Decision Tree

Parameters:


max_depth = 5
min_samples_split = 20


Results:


Training Accuracy: 0.9249
Testing Accuracy: 0.9163


The controlled tree reduced overfitting.

---

# Random Forest Model

Parameters:


n_estimators = 100
max_depth = 10
random_state = 42


Results:


Training Accuracy: 0.9867
Testing Accuracy: 0.9488
ROC-AUC: 0.9905


Top important features:

| Feature | Importance |
|---|---|
| Gr Liv Area | 0.082 |
| Year Built | 0.074 |
| Overall Qual | 0.058 |
| Full Bath | 0.049 |
| Garage Cars | 0.049 |

---

# Gradient Boosting

Parameters:


n_estimators = 100
learning_rate = 0.1
max_depth = 3


Results:


Training Accuracy: 0.9748
Testing Accuracy: 0.9402
ROC-AUC: 0.9901


---

# Feature Ablation Study

The five least important Random Forest features were removed.

Results:

| Model | AUC |
|---|---|
| Full Model | 0.9905 |
| Reduced Model | 0.9912 |

Removing low-importance features slightly improved performance, suggesting these features contributed little useful information.

---

# Cross Validation Comparison

5-fold Stratified Cross Validation was performed.

| Model | Mean AUC | Std |
|---|---|---|
| Logistic Regression | 0.9656 | 0.0041 |
| Decision Tree | 0.9305 | 0.0030 |
| Random Forest | 0.9799 | 0.0041 |
| Gradient Boosting | 0.9821 | 0.0032 |

Gradient Boosting achieved the highest cross-validation AUC.

---

# Hyperparameter Optimization

GridSearchCV was applied on Random Forest.

Search parameters:

- Number of trees
- Maximum depth
- Minimum samples per leaf

Best parameters:

```python
{
'max_depth': None,
'min_samples_leaf': 1,
'n_estimators': 200
}

Best CV AUC:

0.98026
Learning Curve Analysis

Training fractions tested:

20%
40%
60%
80%
100%

The model maintained high training AUC while test AUC increased with more data.

This indicates that collecting more data could further improve performance.

Model Serialization

The best pipeline was saved using:

joblib.dump()

Saved file:

best_model.pkl

The saved model was successfully loaded and tested with new predictions.

Part 4 — LLM Powered Data Intelligence
Objective

The final stage integrates an LLM explanation layer on top of the trained ML model.

Selected Track
Track C — Model Prediction Explanation Pipeline

The system:

Loads the trained model.
Generates predictions.
Generates prediction probabilities.
Sends results to an LLM.
Receives structured explanations.
LLM Integration

API Provider:

Groq API

Model:

llama-3.1-8b-instant

The API key was stored securely using:

.env

No credentials were committed.

Structured Output

The LLM generates:

Prediction label
Confidence level
Primary reason
Secondary reason
Recommended next step

Responses are validated using JSON Schema.

Example Output
{
"prediction_label":"High Price",
"confidence_level":"high",
"top_reason":"High Overall Quality influences prediction",
"second_reason":"Property features indicate higher value",
"next_step":"Verify property details"
}
Temperature Experiment

The model was tested with:

Temperature = 0
Temperature = 0.7

Observation:

Temperature	Result
0	More consistent and deterministic
0.7	More variation in responses

Temperature 0 was selected for structured outputs.

Safety Guardrail

A PII detection layer was implemented.

The system blocks:

Email addresses
Phone numbers

Example:

Input:
example@gmail.com

Output:
Input blocked: PII detected.
