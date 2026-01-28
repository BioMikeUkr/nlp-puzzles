# Module 8: ML Metrics

> Understanding and implementing classification metrics for ML evaluation

## Why This Matters

You can't improve what you don't measure. ML metrics are essential for evaluating model performance, comparing experiments, and making data-driven decisions. Understanding precision vs recall, when to use F1 vs ROC-AUC, and how to interpret confusion matrices is critical for ML engineering roles.

## Key Concepts

### Classification Metrics Overview

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

y_true = [0, 1, 1, 0, 1, 0, 1, 1]
y_pred = [0, 1, 0, 0, 1, 0, 1, 1]

accuracy = accuracy_score(y_true, y_pred)    # 0.875 (7/8 correct)
precision = precision_score(y_true, y_pred)   # 1.0 (no false positives)
recall = recall_score(y_true, y_pred)         # 0.8 (4/5 positives found)
f1 = f1_score(y_true, y_pred)                 # 0.889 (harmonic mean)
```

### Confusion Matrix

**The foundation of all classification metrics:**

```
                Predicted
                0       1
Actual   0     TN      FP
         1     FN      TP
```

- **True Positive (TP):** Correctly predicted positive
- **True Negative (TN):** Correctly predicted negative
- **False Positive (FP):** Incorrectly predicted positive (Type I error)
- **False Negative (FN):** Incorrectly predicted negative (Type II error)

```python
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.ylabel('Actual')
plt.xlabel('Predicted')
```

### Core Metrics

#### Accuracy

**Definition:** Overall correctness
```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

**When to use:**
- Balanced datasets
- All classes equally important

**When NOT to use:**
- Imbalanced datasets (e.g., 99% negative class)
- Different costs for FP vs FN

#### Precision

**Definition:** Of predicted positives, how many are correct?
```
Precision = TP / (TP + FP)
```

**Use case:** Minimize false alarms
- Spam detection (don't mark real emails as spam)
- Fraud detection (don't block legitimate transactions)

**Example:**
```python
# High precision: Few false positives
y_true = [0, 0, 0, 0, 1, 1, 1, 1]
y_pred = [0, 0, 0, 0, 0, 1, 1, 1]  # Precision = 3/3 = 1.0
```

#### Recall (Sensitivity, True Positive Rate)

**Definition:** Of actual positives, how many did we find?
```
Recall = TP / (TP + FN)
```

**Use case:** Don't miss positives
- Cancer detection (don't miss sick patients)
- Security threats (don't miss attacks)

**Example:**
```python
# High recall: Few false negatives
y_true = [0, 0, 0, 0, 1, 1, 1, 1]
y_pred = [1, 0, 0, 0, 1, 1, 1, 1]  # Recall = 4/4 = 1.0
```

#### F1-Score

**Definition:** Harmonic mean of precision and recall
```
F1 = 2 * (Precision * Recall) / (Precision + Recall)
```

**Why harmonic mean?** Punishes extreme values
- If precision = 1.0, recall = 0.1, F1 = 0.18 (not 0.55)
- Requires both metrics to be high

**When to use:**
- Balance precision and recall
- Imbalanced datasets
- Need single metric for optimization

#### F-beta Score

**Definition:** Weighted F-score
```
F_beta = (1 + beta²) * (Precision * Recall) / (beta² * Precision + Recall)
```

**Beta values:**
- **beta = 1:** Standard F1-score (equal weight)
- **beta = 2:** Favor recall (2x weight to recall)
- **beta = 0.5:** Favor precision (2x weight to precision)

```python
from sklearn.metrics import fbeta_score

# F2: Emphasize recall (medical diagnosis)
f2 = fbeta_score(y_true, y_pred, beta=2)

# F0.5: Emphasize precision (spam detection)
f05 = fbeta_score(y_true, y_pred, beta=0.5)
```

### Threshold-Based Metrics

Most classifiers output probabilities. Threshold determines prediction:

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, y_train)

# Get probabilities
y_prob = model.predict_proba(X_test)[:, 1]

# Default threshold = 0.5
y_pred_default = (y_prob >= 0.5).astype(int)

# Custom threshold
threshold = 0.7
y_pred_custom = (y_prob >= threshold).astype(int)
```

#### ROC Curve (Receiver Operating Characteristic)

**What it shows:** Trade-off between TPR and FPR at different thresholds

```python
from sklearn.metrics import roc_curve, roc_auc_score

fpr, tpr, thresholds = roc_curve(y_true, y_prob)
auc = roc_auc_score(y_true, y_prob)

plt.plot(fpr, tpr, label=f'AUC = {auc:.3f}')
plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate (Recall)')
plt.legend()
```

**ROC-AUC Score:**
- 1.0: Perfect classifier
- 0.5: Random classifier
- < 0.5: Worse than random

**When to use ROC-AUC:**
- Compare models across all thresholds
- Balanced datasets
- Care about ranking (not just classification)

#### Precision-Recall Curve

**What it shows:** Trade-off between precision and recall at different thresholds

```python
from sklearn.metrics import precision_recall_curve, average_precision_score

precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
ap = average_precision_score(y_true, y_prob)

plt.plot(recall, precision, label=f'AP = {ap:.3f}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend()
```

**When to use PR curve instead of ROC:**
- Imbalanced datasets
- Care more about positive class
- FP and FN have different costs

### Multi-Class Metrics

**Averaging strategies:**

```python
from sklearn.metrics import classification_report

y_true = [0, 1, 2, 0, 1, 2, 0, 1]
y_pred = [0, 2, 2, 0, 1, 1, 0, 1]

print(classification_report(y_true, y_pred))
```

**Macro average:** Average metrics across classes (equal weight)
```python
# Class 0: P=1.0, R=1.0, F1=1.0
# Class 1: P=1.0, R=1.0, F1=1.0
# Class 2: P=0.5, R=0.5, F1=0.5
# Macro F1 = (1.0 + 1.0 + 0.5) / 3 = 0.83
```

**Weighted average:** Weight by class frequency
```python
# Weighted F1 = (3*1.0 + 3*1.0 + 2*0.5) / 8 = 0.88
```

**Micro average:** Aggregate all TP, FP, FN across classes
```python
# Total TP=7, FP=1, FN=1
# Micro F1 = 2 * 7 / (2*7 + 1 + 1) = 0.875
```

### Regression Metrics

**Mean Absolute Error (MAE):**
```python
from sklearn.metrics import mean_absolute_error

mae = mean_absolute_error(y_true, y_pred)
# Average absolute difference
```

**Mean Squared Error (MSE):**
```python
from sklearn.metrics import mean_squared_error

mse = mean_squared_error(y_true, y_pred)
# Penalizes large errors more
```

**Root Mean Squared Error (RMSE):**
```python
rmse = mean_squared_error(y_true, y_pred, squared=False)
# Same units as target variable
```

**R² Score:**
```python
from sklearn.metrics import r2_score

r2 = r2_score(y_true, y_pred)
# 1.0: Perfect predictions
# 0.0: As good as mean baseline
# < 0.0: Worse than mean baseline
```

## Common Patterns

### Pattern 1: Evaluate Binary Classifier

```python
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

def evaluate_binary_classifier(y_true, y_pred, y_prob=None):
    """Complete evaluation of binary classifier"""

    # Basic metrics
    print(classification_report(y_true, y_pred))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()

    # ROC curve (if probabilities available)
    if y_prob is not None:
        from sklearn.metrics import roc_curve, roc_auc_score

        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)

        plt.plot(fpr, tpr, label=f'AUC = {auc:.3f}')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend()
        plt.show()
```

### Pattern 2: Find Optimal Threshold

```python
from sklearn.metrics import precision_recall_curve

def find_optimal_threshold(y_true, y_prob, metric='f1'):
    """Find threshold that maximizes a metric"""

    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)

    if metric == 'f1':
        f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1])
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[optimal_idx]
        optimal_score = f1_scores[optimal_idx]

    elif metric == 'precision':
        # Find threshold for target recall
        target_recall = 0.8
        idx = np.where(recall >= target_recall)[0][-1]
        optimal_threshold = thresholds[idx]
        optimal_score = precision[idx]

    return optimal_threshold, optimal_score
```

### Pattern 3: Compare Models

```python
def compare_models(models, X_test, y_test):
    """Compare multiple models on same test set"""

    results = []
    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        results.append({
            'Model': name,
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred),
            'Recall': recall_score(y_test, y_pred),
            'F1': f1_score(y_test, y_pred),
            'ROC-AUC': roc_auc_score(y_test, y_prob)
        })

    return pd.DataFrame(results).set_index('Model')
```

### Pattern 4: Cross-Validation with Metrics

```python
from sklearn.model_selection import cross_validate

def cv_with_metrics(model, X, y, cv=5):
    """Cross-validation with multiple metrics"""

    scoring = {
        'accuracy': 'accuracy',
        'precision': 'precision',
        'recall': 'recall',
        'f1': 'f1',
        'roc_auc': 'roc_auc'
    }

    scores = cross_validate(model, X, y, cv=cv, scoring=scoring)

    # Aggregate results
    results = {}
    for metric in scoring.keys():
        key = f'test_{metric}'
        results[metric] = {
            'mean': scores[key].mean(),
            'std': scores[key].std()
        }

    return pd.DataFrame(results).T
```

## Metric Selection Guide

| Scenario | Recommended Metric | Why |
|----------|-------------------|-----|
| Balanced dataset, all errors equal | **Accuracy** | Simple, interpretable |
| Imbalanced dataset | **F1, PR-AUC** | Accounts for class imbalance |
| Minimize false positives | **Precision** | Email spam, recommendations |
| Minimize false negatives | **Recall** | Disease detection, fraud |
| Balance precision/recall | **F1** | General binary classification |
| Ranking matters | **ROC-AUC** | Search, recommendation systems |
| Multi-class balanced | **Macro F1** | Equal importance to all classes |
| Multi-class imbalanced | **Weighted F1** | Account for class frequencies |
| Regression | **RMSE, MAE, R²** | Different error interpretations |

## Documentation & Resources

- [scikit-learn Metrics Guide](https://scikit-learn.org/stable/modules/model_evaluation.html)
- [Precision vs Recall](https://developers.google.com/machine-learning/crash-course/classification/precision-and-recall)
- [ROC and PR Curves](https://machinelearningmastery.com/roc-curves-and-precision-recall-curves-for-classification-in-python/)
- [Imbalanced Learn](https://imbalanced-learn.org/)

## Self-Assessment Checklist

- [ ] I understand the confusion matrix and can calculate metrics from it
- [ ] I know when to use precision vs recall
- [ ] I can interpret F1-score and its trade-offs
- [ ] I understand ROC-AUC and when to use it
- [ ] I know the difference between ROC and PR curves
- [ ] I can choose the right metric for imbalanced data
- [ ] I understand macro vs weighted vs micro averaging
- [ ] I can find optimal classification thresholds

---

## Practice Questions

See [QUESTIONS.md](./QUESTIONS.md) for 30 deep-dive questions with detailed answers covering:
- Architecture & Design (Q1-Q10)
- Implementation & Coding (Q11-Q20)
- Debugging & Troubleshooting (Q21-Q25)
- Trade-offs & Decisions (Q26-Q30)

---

## Additional Resources

- [Google ML Crash Course: Classification](https://developers.google.com/machine-learning/crash-course/classification)
- [Understanding ROC Curves](https://www.dataschool.io/roc-curves-and-auc-explained/)
- [Metrics for Imbalanced Data](https://machinelearningmastery.com/tour-of-evaluation-metrics-for-imbalanced-classification/)
- [scikit-learn Examples](https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html)
