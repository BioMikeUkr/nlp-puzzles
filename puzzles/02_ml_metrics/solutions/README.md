# Solutions - ML Metrics Tasks

This directory contains complete solutions for all three ML metrics tasks.

## Solutions Overview

### Solution 1: Calculate Metrics (`solution_01_calculate_metrics.ipynb`)

**What it covers:**
- Calculating precision, recall, F1, accuracy from confusion matrix manually
- Using sklearn metric functions
- Extracting confusion matrix components (TP, FP, TN, FN)
- Comparing model with naive baselines
- Understanding specificity

**Key code snippets:**

```python
# Manual calculation
precision = TP / (TP + FP)
recall = TP / (TP + FN)
f1 = 2 * (precision * recall) / (precision + recall)
accuracy = (TP + TN) / (TP + TN + FP + FN)

# Using sklearn
from sklearn.metrics import precision_score, recall_score, f1_score
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

# Extract from confusion matrix
cm = confusion_matrix(y_true, y_pred)
tn, fp, fn, tp = cm.ravel()
```

### Solution 2: Threshold Optimization (`solution_02_threshold_optimization.ipynb`)

**What it covers:**
- Finding threshold that maximizes F1-score
- Finding threshold for high precision (>= 95%)
- Finding threshold for high recall (>= 90%)
- Using Youden's J statistic for balanced optimization
- Cost-sensitive threshold selection
- Visualizing threshold effects
- Choosing thresholds for different use cases

**Key code snippets:**

```python
# Find best F1 threshold
thresholds = np.arange(0.1, 0.9, 0.01)
f1_scores = []
for t in thresholds:
    y_pred_t = (y_prob >= t).astype(int)
    f1_scores.append(f1_score(y_true, y_pred_t))
best_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_idx]

# Youden's J statistic
fpr, tpr, roc_thresholds = roc_curve(y_true, y_prob)
j_scores = tpr - fpr
optimal_idx = np.argmax(j_scores)
youden_threshold = roc_thresholds[optimal_idx]

# Cost-sensitive
costs = []
for t in thresholds:
    y_pred_t = (y_prob >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_t).ravel()
    cost = (fp * cost_fp) + (fn * cost_fn)
    costs.append(cost)
optimal_cost_threshold = thresholds[np.argmin(costs)]
```

### Solution 3: Imbalanced Evaluation (`solution_03_imbalanced_evaluation.ipynb`)

**What it covers:**
- Creating and evaluating baseline predictors
- Understanding why accuracy is misleading on imbalanced data
- Comparing ROC-AUC vs PR-AUC
- Analyzing class-wise performance
- Stratifying performance by confidence
- Recommending appropriate metrics

**Key code snippets:**

```python
# Baseline predictors
y_pred_all_0 = np.zeros(len(y_true))  # Always predict majority
y_pred_all_1 = np.ones(len(y_true))   # Always predict minority
y_pred_random = np.random.choice([0, 1], size=len(y_true), p=[0.8, 0.2])

# Compare ROC-AUC vs PR-AUC
roc_auc = roc_auc_score(y_true, y_prob)
pr_auc = average_precision_score(y_true, y_prob)

# Class-wise analysis
for class_label in [0, 1]:
    mask = (y_true == class_label)
    class_accuracy = (y_true[mask] == y_pred[mask]).mean()

# Stratify by confidence
high_conf = (y_prob > 0.7) | (y_prob < 0.3)
low_conf = (y_prob >= 0.3) & (y_prob <= 0.7)
high_acc = accuracy_score(y_true[high_conf], y_pred[high_conf])
low_acc = accuracy_score(y_true[low_conf], y_pred[low_conf])
```

## How to Use Solutions

### Before Looking at Solutions:

1. **Attempt the task yourself** - Try to complete all code cells
2. **Run the tests** - See which assertions fail
3. **Review learning notebooks** - Go back to relevant sections
4. **Debug your approach** - Use print statements to understand errors

### When Using Solutions:

1. **Don't copy-paste blindly** - Understand the logic
2. **Compare approaches** - Your solution might be different but correct
3. **Learn from differences** - See if solution is more efficient/elegant
4. **Note the explanations** - Solutions include markdown explaining "why"

### After Reviewing Solutions:

1. **Reattempt from scratch** - Close solution and try again
2. **Apply to new data** - Test understanding with different datasets
3. **Extend the solution** - Add visualizations or additional metrics
4. **Teach someone else** - Best way to solidify understanding

## Solution Approaches

### Task 1: Manual Calculation Strategy

```python
# Step 1: Understand what you're calculating
# Precision = "Of positive predictions, how many are correct?"
# Recall = "Of actual positives, how many did we find?"

# Step 2: Extract components
TP = ((y_true == 1) & (y_pred == 1)).sum()
FP = ((y_true == 0) & (y_pred == 1)).sum()
FN = ((y_true == 1) & (y_pred == 0)).sum()
TN = ((y_true == 0) & (y_pred == 0)).sum()

# Step 3: Calculate metrics
precision = TP / (TP + FP) if (TP + FP) > 0 else 0
recall = TP / (TP + FN) if (TP + FN) > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

# Step 4: Verify with sklearn
assert abs(precision - precision_score(y_true, y_pred)) < 0.001
```

### Task 2: Threshold Search Strategy

```python
# Step 1: Define search space
thresholds = np.linspace(0.1, 0.9, 100)  # 100 thresholds

# Step 2: Calculate metric for each threshold
metric_values = []
for threshold in thresholds:
    y_pred_threshold = (y_prob >= threshold).astype(int)
    metric = your_metric_function(y_true, y_pred_threshold)
    metric_values.append(metric)

# Step 3: Find optimal
optimal_idx = np.argmax(metric_values)  # or np.argmin for cost
optimal_threshold = thresholds[optimal_idx]

# Step 4: For constrained optimization (e.g., precision >= 0.95)
valid_indices = [i for i, p in enumerate(precisions) if p >= 0.95]
if valid_indices:
    best_idx = valid_indices[np.argmax([recalls[i] for i in valid_indices])]
    optimal_threshold = thresholds[best_idx]
```

### Task 3: Imbalanced Data Strategy

```python
# Step 1: Establish baselines
baseline_all_majority = np.zeros(len(y_true))
baseline_accuracy = accuracy_score(y_true, baseline_all_majority)

# Step 2: Calculate multiple metrics
metrics = {
    'accuracy': accuracy_score(y_true, y_pred),
    'precision': precision_score(y_true, y_pred),
    'recall': recall_score(y_true, y_pred),
    'f1': f1_score(y_true, y_pred),
    'roc_auc': roc_auc_score(y_true, y_prob),
    'pr_auc': average_precision_score(y_true, y_prob)
}

# Step 3: Compare improvements
improvements = {
    k: v - baseline_metrics.get(k, 0)
    for k, v in metrics.items()
}

# Step 4: Analyze which metrics are most informative
# F1 and PR-AUC will show larger improvements than accuracy
```

## Common Mistakes and Solutions

### Mistake 1: Dividing by zero
```python
# Wrong:
precision = TP / (TP + FP)  # Fails if no positive predictions

# Right:
precision = TP / (TP + FP) if (TP + FP) > 0 else 0
# Or use sklearn with zero_division parameter
precision = precision_score(y_true, y_pred, zero_division=0)
```

### Mistake 2: Wrong confusion matrix interpretation
```python
# sklearn confusion matrix layout:
# [[TN, FP],
#  [FN, TP]]

# Not:
# [[TP, FP],
#  [FN, TN]]

# Safe extraction:
cm = confusion_matrix(y_true, y_pred)
tn, fp, fn, tp = cm.ravel()
```

### Mistake 3: Forgetting to convert probabilities to predictions
```python
# Wrong:
f1 = f1_score(y_true, y_prob)  # y_prob is continuous

# Right:
y_pred = (y_prob >= threshold).astype(int)
f1 = f1_score(y_true, y_pred)
```

### Mistake 4: Not handling edge cases
```python
# Consider:
# - All predictions are one class
# - Very few samples of one class
# - Extreme thresholds (0.0 or 1.0)

# Use try-except or check for edge cases:
if len(np.unique(y_pred)) == 1:
    print("Warning: All predictions are the same class")
```

## Testing Your Solutions

Each solution notebook includes test assertions. To verify correctness:

```python
# Tests will check:
1. Variable is not None
2. Variable is correct type (int, float, array, DataFrame)
3. Values are in valid range (e.g., 0-1 for probabilities)
4. Calculations match expected results
5. Logic is correct (e.g., high precision threshold > F1 threshold)

# Example test:
assert precision is not None, "Calculate precision"
assert 0 <= precision <= 1, "Precision must be between 0 and 1"
assert abs(precision - expected) < 0.001, f"Expected {expected}, got {precision}"
```

## Performance Optimization Tips

For large datasets, optimize threshold search:

```python
# Instead of:
for threshold in np.linspace(0, 1, 1000):
    y_pred = (y_prob >= threshold).astype(int)
    metric = calculate_metric(y_true, y_pred)

# Use precision_recall_curve or roc_curve:
precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
# These return all unique threshold values efficiently
```

## Extending the Solutions

Try these challenges:

1. **Add visualizations** - Plot confusion matrices, threshold curves
2. **Try different datasets** - Load multiclass_data.csv
3. **Implement custom metrics** - Matthews Correlation Coefficient, Cohen's Kappa
4. **Build a dashboard** - Interactive threshold selection with Plotly
5. **Compare multiple models** - Load different probability predictions
6. **Cross-validation** - Calculate metrics with CV
7. **Bootstrap confidence intervals** - Estimate metric uncertainty

## Additional Resources

- **sklearn metrics guide**: https://scikit-learn.org/stable/modules/model_evaluation.html
- **Imbalanced learning**: https://imbalanced-learn.org/
- **Metric selection guide**: ../README.md
- **Conceptual Q&A**: ../QUESTIONS.md

## Next Steps

After completing solutions:

1. Attempt to solve tasks without looking at solutions
2. Apply these techniques to your own datasets
3. Read QUESTIONS.md for deeper understanding
4. Move to Module 9 (MLOps) to learn metric tracking with MLflow
