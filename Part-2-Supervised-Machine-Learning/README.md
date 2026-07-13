# Part 2 — Supervised Machine Learning Model

## Objective

This section focuses on building and evaluating supervised machine learning models using the cleaned dataset generated in Part 1.

Two tasks are performed:

1. Regression:
   - Predict continuous house prices using `SalePrice`

2. Classification:
   - Predict whether a house price is above or below the median value

---

# Dataset and Target Definition

## Regression Target

The regression target is:

```
y_reg = SalePrice
```

The model predicts the actual house price value.

---

## Classification Target

A binary classification target was created by converting `SalePrice` using its median value:

```python
y_clf = (SalePrice > SalePrice.median()).astype(int)
```

Class labels:

- `0` → SalePrice below or equal to median
- `1` → SalePrice above median

---

# Data Preprocessing

## Categorical Encoding

Categorical features were converted into numerical form using One-Hot Encoding.

Implementation:

```python
pd.get_dummies(drop_first=True)
```

`drop_first=True` was used to reduce multicollinearity by removing one redundant dummy variable.

One-hot encoding was selected because categorical values do not have a natural numerical order.

---

## Train-Test Split and Scaling

The dataset was split using:

```python
test_size = 0.2
random_state = 42
```

Feature scaling was performed using:

```python
StandardScaler()
```

The scaler was fitted only on training data to prevent data leakage.

---

# Regression Models

## Linear Regression

A Linear Regression model was trained to predict `SalePrice`.

Evaluation metrics:

- Mean Squared Error (MSE)
- R² Score

---

## Ridge Regression

Ridge Regression was implemented using:

```python
Ridge(alpha=1.0)
```

Ridge applies L2 regularization to reduce large coefficients and improve generalization.

---

## Regression Results

| Model | MSE | R² Score |
|---|---|---|
| Linear Regression | 8.341823e+08 | 0.895955 |
| Ridge Regression | 8.298233e+08 | 0.896499 |

Ridge Regression slightly improved performance by reducing MSE and increasing the R² score.

---

# Coefficient Analysis

The features with the highest absolute coefficients were:

| Feature | Coefficient |
|---|---|
| BsmtFin SF 1 | 133328.89 |
| Bsmt Unf SF | 120065.28 |
| Total Bsmt SF | -112078.08 |

Since features were standardized, coefficients represent the effect of a one standard deviation increase in the feature.

- Positive coefficient → associated with an increase in predicted SalePrice.
- Negative coefficient → associated with a decrease in predicted SalePrice.

---

# Classification Model

## Logistic Regression

A Logistic Regression model was trained with:

```python
LogisticRegression(max_iter=1000)
```

The classification target was balanced:

```
Class 0: 1467
Class 1: 1463
```

Since both classes contained approximately equal samples, no imbalance handling technique was required.

---

# Classification Evaluation

The model was evaluated using:

- Confusion Matrix
- Accuracy
- Precision
- Recall
- F1 Score
- ROC Curve
- AUC Score

Precision formula:

```
Precision = TP / (TP + FP)
```

Recall formula:

```
Recall = TP / (TP + FN)
```

---

# Decision Threshold Analysis

Different decision thresholds were tested:

```
0.30, 0.40, 0.50, 0.60, 0.70
```

The best F1-score was achieved at:

```
Threshold = 0.50
```

Results:

```
Precision = 0.9433
Recall = 0.9279
F1 Score = 0.9355
```

Changing the threshold creates a trade-off:

- Higher threshold → higher precision, lower recall
- Lower threshold → higher recall, more false positives

---

# Regularization Experiment

Two Logistic Regression models were compared:

| Model | C Value |
|---|---|
| Baseline Logistic Regression | C = 1.0 |
| Strong Regularization | C = 0.01 |

The parameter `C` controls the inverse strength of regularization.

Lower C values apply stronger L2 regularization.

---

# Bootstrap Confidence Interval

Bootstrap testing was performed using:

```
500 bootstrap samples
```

The AUC difference was calculated:

```
AUC(C=1.0) - AUC(C=0.01)
```

Results:

```
Mean AUC Difference:
-0.01156

95% Confidence Interval:
[-0.01965, -0.00547]
```

The confidence interval does not include zero, indicating that the performance difference is consistent across different samples.

The negative difference indicates that the C=0.01 model achieved slightly better AUC performance.

---

# Files

```
Part-2-Supervised-Machine-Learning/

│
├── Part2_Model_Training.ipynb
└── README.md
```

---

# Libraries Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
