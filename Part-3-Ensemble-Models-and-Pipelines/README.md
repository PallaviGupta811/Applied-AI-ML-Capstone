# Part 3 — Advanced Modeling: Ensembles, Tuning and ML Pipeline

## Objective

This section focuses on improving model performance using advanced machine learning techniques.

The tasks include:

- Decision Tree optimization
- Ensemble learning
- Feature importance analysis
- Feature ablation
- Cross-validation comparison
- Hyperparameter tuning
- Model serialization using pipelines

---

# Dataset Preparation

The cleaned dataset from Part 1 was used.

The classification target was created from `SalePrice`:

```python
y_clf = (SalePrice > SalePrice.median()).astype(int)
```

Classes:

- `0` → House price below or equal to median
- `1` → House price above median

Categorical features were converted using one-hot encoding.

Feature scaling was performed using `StandardScaler`.

---

# 1. Decision Tree Baseline

A default Decision Tree classifier was trained:

```python
DecisionTreeClassifier()
```

Results:

| Model | Training Accuracy | Testing Accuracy |
|---|---|---|
| Decision Tree | 1.0000 | 0.9096 |

The model shows signs of overfitting because it achieves perfect training accuracy but lower test accuracy.

Decision trees are high-variance models because they make greedy splits and can memorize training patterns.

---

# 2. Controlled Decision Tree

A restricted decision tree was trained:

```python
max_depth = 5
min_samples_split = 20
```

Results:

| Model | Training Accuracy | Testing Accuracy |
|---|---|---|
| Controlled Decision Tree | 0.9249 | 0.9164 |

The controlled tree reduced overfitting by limiting tree complexity.

- `max_depth` controls how deep the tree can grow.
- `min_samples_split` prevents creating splits based on very small noisy groups.

Compared to the baseline tree, the train-test accuracy gap decreased significantly.

---

# 3. Gini vs Entropy Comparison

Two decision trees were trained using different splitting criteria.

Results:

| Criterion | Test Accuracy |
|---|---|
| Gini | Completed in notebook |
| Entropy | Completed in notebook |

### Gini Impurity Formula

\[
Gini = 1 - \sum p_i^2
\]

A Gini value of 0 means all samples in the node belong to the same class.

### Entropy Formula

\[
Entropy = -\sum p_i \log_2(p_i)
\]

Entropy measures the uncertainty or disorder in a node.

---

# 4. Random Forest Classifier

A Random Forest model was trained:

```python
RandomForestClassifier(
    n_estimators=100,
    max_depth=10
)
```

Results:

| Metric | Score |
|---|---|
| Training Accuracy | 0.9868 |
| Testing Accuracy | 0.9488 |
| ROC-AUC | 0.9905 |

Random Forest reduces variance by combining multiple decision trees.

Each tree is trained on bootstrap samples and considers a random subset of features during splitting.

---

# Random Forest Feature Importance

Top 5 important features:

| Feature | Importance |
|---|---|
| Gr Liv Area | 0.082375 |
| Year Built | 0.074415 |
| Overall Qual | 0.058056 |
| Full Bath | 0.049611 |
| Garage Cars | 0.049173 |

Feature importance represents the average reduction in impurity contributed by a feature across all trees.

Unlike linear regression coefficients, feature importance does not represent the direction of impact, only the contribution to prediction.

---

# 5. Gradient Boosting Classifier

Gradient Boosting was trained with:

```python
n_estimators=100
learning_rate=0.1
max_depth=3
```

Results:

| Metric | Score |
|---|---|
| Training Accuracy | 0.9748 |
| Testing Accuracy | 0.9403 |
| ROC-AUC | 0.9902 |

Gradient Boosting builds trees sequentially, where each new tree attempts to correct errors from previous trees.

---

# 6. Feature Ablation Study

The five least important Random Forest features were removed.

Removed features:

- Kitchen Qual_Po
- MS Zoning_I (all)
- Garage Qual_Po
- Garage Cond_Po
- Sale Condition_AdjLand


Comparison:

| Model | Number of Features | Test AUC |
|---|---|---|
| Full Random Forest | 262 | 0.9905 |
| Reduced Random Forest | 257 | 0.9912 |

Removing low-importance features slightly improved performance.

This suggests these features were adding noise rather than useful information.

A lower-dimensional model can reduce:

- Inference cost
- Memory usage
- Maintenance complexity

while maintaining similar performance.

---

# 7. Cross Validation Comparison

5-fold Stratified Cross Validation was performed.

| Model | Mean AUC | Std AUC |
|---|---|---|
| Logistic Regression | 0.9656 | 0.0041 |
| Decision Tree | 0.9305 | 0.0030 |
| Random Forest | 0.9799 | 0.0041 |
| Gradient Boosting | 0.9821 | 0.0033 |

Cross-validation provides a more reliable estimate of model performance because the model is evaluated across multiple train-validation splits.

---

# 8. Hyperparameter Tuning with GridSearchCV

A Random Forest pipeline was created:

Pipeline:

```
SimpleImputer
      ↓
StandardScaler
      ↓
RandomForestClassifier
```

Parameter grid:

```python
n_estimators:
[50,100,200]

max_depth:
[5,10,None]

min_samples_leaf:
[1,5]
```

Total configurations evaluated:

```
3 × 3 × 2 = 18 models

18 × 5 folds = 90 total fits
```

---

## Best Parameters

```text
max_depth = None

min_samples_leaf = 1

n_estimators = 200
```

Best Cross Validation AUC:

```
0.98026
```

Grid Search performs exhaustive searching, while Randomized Search can explore large spaces faster by testing a fixed number of random combinations.

---

# 9. Learning Curve Analysis

The tuned pipeline was trained on increasing training data sizes.

| Training Fraction | Training AUC | Test AUC |
|---|---|---|
| 0.2 | 1.0000 | 0.9878 |
| 0.4 | 1.0000 | 0.9888 |
| 0.6 | 1.0000 | 0.9892 |
| 0.8 | 1.0000 | 0.9901 |
| 1.0 | 1.0000 | 0.9911 |

The training AUC remains high because Random Forest can fit training data very well.

The test AUC increases as more training data is added, indicating that the model is currently more data-limited than capacity-limited.

---

# 10. Model Serialization

The best GridSearchCV pipeline was saved using:

```python
joblib.dump()
```

Saved model:

```
best_model.pkl
```

The saved pipeline was successfully reloaded and tested using:

```python
joblib.load()
```

Predictions were generated successfully.

---

# Final Model Recommendation

Based on the experiments, Gradient Boosting achieved the highest cross-validation AUC, while Random Forest achieved excellent test performance and feature interpretability.

The tuned Random Forest pipeline is recommended because it provides strong performance, robustness, and can be easily deployed using the serialized pipeline.

The final pipeline balances predictive performance with reproducibility and production readiness.

---

# Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Joblib
- Jupyter Notebook