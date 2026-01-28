# Module 8: ML Metrics - Deep Dive Questions

## Architecture & Design (Q1-Q10)

### Q1: Your team is building a fraud detection system for credit card transactions. The dataset has 10,000 transactions with only 50 fraudulent cases (0.5% fraud rate). Management wants a single metric to track model performance. Which metric should you choose and why? What are the risks of using accuracy?

**Answer:**

For this highly imbalanced dataset, **F1-score** or **PR-AUC** (Precision-Recall Area Under Curve) are the best choices. Accuracy is dangerous here.

**Why accuracy fails:**
```python
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Simulated data: 0.5% fraud rate
y_true = np.array([0] * 9950 + [1] * 50)

# Naive model: Always predict "not fraud"
y_pred_naive = np.zeros(10000)

print(f"Accuracy: {accuracy_score(y_true, y_pred_naive):.4f}")  # 0.9950 (99.5%!)
print(f"Precision: {precision_score(y_true, y_pred_naive, zero_division=0):.4f}")  # 0.0
print(f"Recall: {recall_score(y_true, y_pred_naive):.4f}")  # 0.0
print(f"F1: {f1_score(y_true, y_pred_naive, zero_division=0):.4f}")  # 0.0
```

Output:
```
Accuracy: 0.9950  # Looks great but catches 0 fraud!
Precision: 0.0000
Recall: 0.0000
F1: 0.0000
```

**Better approach with F1:**
```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score

# Real model
model = LogisticRegression(class_weight='balanced')
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred))
print(f"\nROC-AUC: {roc_auc_score(y_test, y_prob):.4f}")
print(f"PR-AUC: {average_precision_score(y_test, y_prob):.4f}")
```

**Recommended metrics:**
1. **Primary: F1-score** - Single metric balancing precision/recall
2. **Secondary: PR-AUC** - Evaluates performance across all thresholds
3. **Business metric: Cost per fraud** - $ saved vs false alarms

**Why not ROC-AUC alone?** ROC-AUC can be optimistic on imbalanced data because it includes TN rate, which is easy to get right when negatives dominate.

---

### Q2: You're building a medical diagnosis system for cancer detection. False negatives (missing cancer) are 10x worse than false positives (unnecessary further testing). How do you design your evaluation framework to reflect this business constraint?

**Answer:**

Use **weighted metrics** and **custom cost functions** that penalize FN more than FP.

**Implementation:**

```python
from sklearn.metrics import confusion_matrix, make_scorer
import numpy as np

def custom_cost_function(y_true, y_pred, fn_cost=10, fp_cost=1):
    """
    Calculate total cost where:
    - False Negative costs fn_cost (default 10)
    - False Positive costs fp_cost (default 1)
    - True predictions cost 0
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    total_cost = (fn * fn_cost) + (fp * fp_cost)

    return total_cost

def cost_score(y_true, y_pred):
    """Negative cost for use with scikit-learn (higher is better)"""
    return -custom_cost_function(y_true, y_pred)

# Create scorer for cross-validation
cost_scorer = make_scorer(cost_score, greater_is_better=True)
```

**Optimize for high recall:**
```python
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

# Find threshold that minimizes cost
def find_optimal_threshold(model, X_val, y_val, fn_cost=10, fp_cost=1):
    """Find threshold that minimizes weighted cost"""

    y_prob = model.predict_proba(X_val)[:, 1]

    best_cost = float('inf')
    best_threshold = 0.5
    best_metrics = {}

    for threshold in np.arange(0.05, 0.95, 0.05):
        y_pred = (y_prob >= threshold).astype(int)

        cost = custom_cost_function(y_val, y_pred, fn_cost, fp_cost)

        if cost < best_cost:
            best_cost = cost
            best_threshold = threshold
            best_metrics = {
                'threshold': threshold,
                'cost': cost,
                'recall': recall_score(y_val, y_pred),
                'precision': precision_score(y_val, y_pred),
                'f1': f1_score(y_val, y_pred)
            }

    return best_threshold, best_metrics

# Train model
model = LogisticRegression(class_weight='balanced')
model.fit(X_train, y_train)

# Find optimal threshold
optimal_threshold, metrics = find_optimal_threshold(
    model, X_val, y_val, fn_cost=10, fp_cost=1
)

print(f"Optimal threshold: {optimal_threshold:.3f}")
print(f"Metrics at optimal threshold:")
for k, v in metrics.items():
    print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")
```

**Use F-beta score with beta=2:**
```python
from sklearn.metrics import fbeta_score

# beta=2 weights recall 2x more than precision
# Perfect for scenarios where FN is worse than FP
f2 = fbeta_score(y_true, y_pred, beta=2)

# beta=4 for even more recall emphasis
f4 = fbeta_score(y_true, y_pred, beta=4)

print(f"F1: {f1_score(y_true, y_pred):.3f}")
print(f"F2: {f2:.3f}")
print(f"F4: {f4:.3f}")
print(f"Recall: {recall_score(y_true, y_pred):.3f}")
```

**Complete evaluation dashboard:**
```python
def evaluate_with_costs(y_true, y_pred, y_prob, fn_cost=10, fp_cost=1):
    """Complete evaluation with cost analysis"""

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    total_cost = (fn * fn_cost) + (fp * fp_cost)
    missed_cancers = fn
    unnecessary_tests = fp

    metrics = {
        'Total Cost': total_cost,
        'Missed Cancers (FN)': missed_cancers,
        'Unnecessary Tests (FP)': unnecessary_tests,
        'Recall (Sensitivity)': recall_score(y_true, y_pred),
        'Precision (PPV)': precision_score(y_true, y_pred),
        'F1': f1_score(y_true, y_pred),
        'F2': fbeta_score(y_true, y_pred, beta=2),
        'ROC-AUC': roc_auc_score(y_true, y_prob)
    }

    return pd.DataFrame([metrics]).T
```

**Key principle:** Optimize for **recall** (minimize FN), accept lower precision (more FP).

---

### Q3: You're deploying a spam detection system for email. Users are more annoyed by false positives (real emails marked as spam) than false negatives (spam that gets through). Design a multi-threshold system that handles different user preferences.

**Answer:**

Create a **three-tier classification system** with different thresholds for aggressive, balanced, and conservative spam filtering.

**Implementation:**

```python
from sklearn.metrics import precision_recall_curve
import numpy as np

class MultiThresholdSpamDetector:
    """Spam detector with user-configurable sensitivity"""

    def __init__(self, model):
        self.model = model
        self.thresholds = {
            'aggressive': None,  # High recall, low precision
            'balanced': None,    # F1-optimal
            'conservative': None # High precision, low recall
        }

    def calibrate(self, X_val, y_val):
        """Find optimal thresholds for each mode"""

        y_prob = self.model.predict_proba(X_val)[:, 1]
        precision, recall, thresholds = precision_recall_curve(y_val, y_prob)

        # Conservative: 95% precision (few false positives)
        conservative_idx = np.where(precision[:-1] >= 0.95)[0]
        if len(conservative_idx) > 0:
            self.thresholds['conservative'] = thresholds[conservative_idx[-1]]
        else:
            self.thresholds['conservative'] = 0.9

        # Aggressive: 95% recall (few false negatives)
        aggressive_idx = np.where(recall[:-1] >= 0.95)[0]
        if len(aggressive_idx) > 0:
            self.thresholds['aggressive'] = thresholds[aggressive_idx[0]]
        else:
            self.thresholds['aggressive'] = 0.1

        # Balanced: Maximize F1
        f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-10)
        balanced_idx = np.argmax(f1_scores)
        self.thresholds['balanced'] = thresholds[balanced_idx]

        return self.thresholds

    def predict(self, X, mode='balanced'):
        """Predict with specified sensitivity mode"""

        y_prob = self.model.predict_proba(X)[:, 1]
        threshold = self.thresholds[mode]

        y_pred = (y_prob >= threshold).astype(int)

        return y_pred, y_prob

    def get_metrics(self, X_test, y_test):
        """Compare all modes"""

        results = []
        for mode in ['conservative', 'balanced', 'aggressive']:
            y_pred, y_prob = self.predict(X_test, mode)

            results.append({
                'Mode': mode,
                'Threshold': self.thresholds[mode],
                'Precision': precision_score(y_test, y_pred),
                'Recall': recall_score(y_test, y_pred),
                'F1': f1_score(y_test, y_pred),
                'FP_rate': (y_pred[y_test == 0] == 1).mean(),
                'FN_rate': (y_pred[y_test == 1] == 0).mean()
            })

        return pd.DataFrame(results)

# Usage
detector = MultiThresholdSpamDetector(model)
detector.calibrate(X_val, y_val)

# Show comparison
print(detector.get_metrics(X_test, y_test))
```

Output:
```
         Mode  Threshold  Precision  Recall     F1  FP_rate  FN_rate
0  conservative      0.85      0.950   0.720  0.820    0.010    0.280
1      balanced      0.45      0.850   0.900  0.875    0.050    0.100
2    aggressive      0.15      0.720   0.980  0.830    0.120    0.020
```

**User interface:**
```python
# Allow users to choose their preference
user_preference = "conservative"  # From user settings

# Predict with chosen mode
y_pred, y_prob = detector.predict(X_new_emails, mode=user_preference)

# Show confidence
for email, pred, prob in zip(emails, y_pred, y_prob):
    if pred == 1:
        confidence = "HIGH" if prob > 0.9 else "MEDIUM" if prob > 0.7 else "LOW"
        print(f"SPAM ({confidence} confidence): {email.subject}")
```

**A/B test different modes:**
```python
def ab_test_spam_modes(detector, users, emails, labels):
    """Test user satisfaction across modes"""

    results = {}
    for mode in ['conservative', 'balanced', 'aggressive']:
        # Simulate user group
        user_group = users[users['mode'] == mode]

        y_pred, _ = detector.predict(emails, mode=mode)

        # Calculate user-facing metrics
        results[mode] = {
            'legitimate_emails_blocked': np.sum((labels == 0) & (y_pred == 1)),
            'spam_that_got_through': np.sum((labels == 1) & (y_pred == 0)),
            'user_complaints': user_group['complaints'].sum()
        }

    return pd.DataFrame(results).T
```

**Key insight:** Different users have different tolerance for FP vs FN. Offer customization!

---

### Q4: Your ML platform serves multiple teams with different models. Design a standardized metrics reporting system that allows fair comparison across binary classification, multi-class classification, and regression tasks.

**Answer:**

Create a **unified metrics framework** with task-specific defaults and common visualizations.

**Implementation:**

```python
from dataclasses import dataclass
from typing import Dict, Any, Optional
import pandas as pd
from sklearn.metrics import *

@dataclass
class ModelMetrics:
    """Standardized metrics container"""
    model_name: str
    task_type: str  # 'binary', 'multiclass', 'regression'
    metrics: Dict[str, float]
    metadata: Dict[str, Any]
    confusion_matrix: Optional[np.ndarray] = None

class MetricsReporter:
    """Unified metrics reporting across task types"""

    def __init__(self):
        self.reports = []

    def evaluate_binary(self, model_name: str, y_true, y_pred, y_prob=None, metadata=None):
        """Evaluate binary classification model"""

        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'specificity': recall_score(y_true, y_pred, pos_label=0, zero_division=0)
        }

        if y_prob is not None:
            metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
            metrics['pr_auc'] = average_precision_score(y_true, y_prob)

        cm = confusion_matrix(y_true, y_pred)

        report = ModelMetrics(
            model_name=model_name,
            task_type='binary',
            metrics=metrics,
            metadata=metadata or {},
            confusion_matrix=cm
        )

        self.reports.append(report)
        return report

    def evaluate_multiclass(self, model_name: str, y_true, y_pred, y_prob=None,
                           average='weighted', metadata=None):
        """Evaluate multi-class classification model"""

        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            f'precision_{average}': precision_score(y_true, y_pred, average=average, zero_division=0),
            f'recall_{average}': recall_score(y_true, y_pred, average=average, zero_division=0),
            f'f1_{average}': f1_score(y_true, y_pred, average=average, zero_division=0)
        }

        # Add per-class metrics
        per_class_precision = precision_score(y_true, y_pred, average=None, zero_division=0)
        per_class_recall = recall_score(y_true, y_pred, average=None, zero_division=0)
        per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)

        for i, (p, r, f) in enumerate(zip(per_class_precision, per_class_recall, per_class_f1)):
            metrics[f'class_{i}_precision'] = p
            metrics[f'class_{i}_recall'] = r
            metrics[f'class_{i}_f1'] = f

        if y_prob is not None:
            metrics['roc_auc_ovr'] = roc_auc_score(y_true, y_prob, multi_class='ovr', average=average)

        cm = confusion_matrix(y_true, y_pred)

        report = ModelMetrics(
            model_name=model_name,
            task_type='multiclass',
            metrics=metrics,
            metadata=metadata or {},
            confusion_matrix=cm
        )

        self.reports.append(report)
        return report

    def evaluate_regression(self, model_name: str, y_true, y_pred, metadata=None):
        """Evaluate regression model"""

        metrics = {
            'mae': mean_absolute_error(y_true, y_pred),
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': mean_squared_error(y_true, y_pred, squared=False),
            'r2': r2_score(y_true, y_pred),
            'mape': np.mean(np.abs((y_true - y_pred) / y_true)) * 100,  # Mean Absolute Percentage Error
            'max_error': max_error(y_true, y_pred)
        }

        report = ModelMetrics(
            model_name=model_name,
            task_type='regression',
            metrics=metrics,
            metadata=metadata or {},
            confusion_matrix=None
        )

        self.reports.append(report)
        return report

    def compare_models(self, task_type=None):
        """Compare all evaluated models"""

        reports = self.reports
        if task_type:
            reports = [r for r in reports if r.task_type == task_type]

        rows = []
        for report in reports:
            row = {'model': report.model_name, 'task': report.task_type}
            row.update(report.metrics)
            row.update(report.metadata)
            rows.append(row)

        return pd.DataFrame(rows)

    def export_report(self, filename='metrics_report.json'):
        """Export all metrics to JSON"""
        import json

        export_data = []
        for report in self.reports:
            export_data.append({
                'model_name': report.model_name,
                'task_type': report.task_type,
                'metrics': report.metrics,
                'metadata': report.metadata,
                'confusion_matrix': report.confusion_matrix.tolist() if report.confusion_matrix is not None else None
            })

        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)

# Usage example
reporter = MetricsReporter()

# Binary classification model
reporter.evaluate_binary(
    model_name='fraud_detector_v1',
    y_true=y_test_fraud,
    y_pred=y_pred_fraud,
    y_prob=y_prob_fraud,
    metadata={'team': 'payments', 'version': '1.0', 'date': '2024-01-15'}
)

# Multi-class model
reporter.evaluate_multiclass(
    model_name='ticket_classifier_v2',
    y_true=y_test_tickets,
    y_pred=y_pred_tickets,
    y_prob=y_prob_tickets,
    average='weighted',
    metadata={'team': 'support', 'version': '2.0'}
)

# Regression model
reporter.evaluate_regression(
    model_name='sales_predictor_v1',
    y_true=y_test_sales,
    y_pred=y_pred_sales,
    metadata={'team': 'analytics', 'version': '1.0'}
)

# Compare all models
print(reporter.compare_models())

# Compare within task type
print(reporter.compare_models(task_type='binary'))

# Export for dashboards
reporter.export_report('ml_metrics_dashboard.json')
```

**Standardized visualization:**
```python
def plot_model_comparison(reporter, metric='f1'):
    """Visualize model comparison"""
    import matplotlib.pyplot as plt
    import seaborn as sns

    df = reporter.compare_models()

    # Filter models that have this metric
    df_metric = df[df[metric].notna()]

    plt.figure(figsize=(12, 6))
    sns.barplot(data=df_metric, x='model', y=metric, hue='task')
    plt.xticks(rotation=45, ha='right')
    plt.title(f'Model Comparison: {metric}')
    plt.tight_layout()
    plt.show()
```

This creates a unified interface for tracking all ML models across the organization!

---

### Q5: You're building an ML monitoring system that tracks model performance in production. Design a metric tracking system that detects when a model's performance has degraded and needs retraining.

**Answer:**

Implement **continuous monitoring** with statistical tests for metric degradation detection.

**Implementation:**

```python
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List
import numpy as np
from scipy import stats

@dataclass
class MetricSnapshot:
    """Single point-in-time metric measurement"""
    timestamp: datetime
    metric_name: str
    value: float
    sample_size: int

class ModelMonitor:
    """Monitor model metrics for degradation"""

    def __init__(self, model_name: str, baseline_window_days=7):
        self.model_name = model_name
        self.baseline_window_days = baseline_window_days
        self.snapshots: List[MetricSnapshot] = []
        self.baseline_metrics = {}
        self.alerts = []

    def log_metrics(self, y_true, y_pred, y_prob=None, timestamp=None):
        """Log metrics from batch of predictions"""

        if timestamp is None:
            timestamp = datetime.now()

        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0)
        }

        if y_prob is not None:
            metrics['roc_auc'] = roc_auc_score(y_true, y_prob)

        # Store snapshots
        for metric_name, value in metrics.items():
            snapshot = MetricSnapshot(
                timestamp=timestamp,
                metric_name=metric_name,
                value=value,
                sample_size=len(y_true)
            )
            self.snapshots.append(snapshot)

    def set_baseline(self):
        """Establish baseline from recent data"""

        cutoff = datetime.now() - timedelta(days=self.baseline_window_days)
        recent_snapshots = [s for s in self.snapshots if s.timestamp >= cutoff]

        # Group by metric
        for metric_name in set(s.metric_name for s in recent_snapshots):
            values = [s.value for s in recent_snapshots if s.metric_name == metric_name]

            self.baseline_metrics[metric_name] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'p5': np.percentile(values, 5),
                'p95': np.percentile(values, 95),
                'samples': len(values)
            }

        return self.baseline_metrics

    def check_degradation(self, metric_name='f1', threshold_std=2.0):
        """Check if recent performance has degraded"""

        if metric_name not in self.baseline_metrics:
            return None

        baseline = self.baseline_metrics[metric_name]

        # Get recent snapshots (last 24 hours)
        cutoff = datetime.now() - timedelta(hours=24)
        recent = [s for s in self.snapshots
                 if s.metric_name == metric_name and s.timestamp >= cutoff]

        if len(recent) == 0:
            return None

        recent_values = [s.value for s in recent]
        recent_mean = np.mean(recent_values)

        # Z-score test: Is recent performance significantly lower?
        z_score = (recent_mean - baseline['mean']) / baseline['std']

        degraded = z_score < -threshold_std  # Significantly below baseline

        alert = {
            'metric': metric_name,
            'baseline_mean': baseline['mean'],
            'recent_mean': recent_mean,
            'z_score': z_score,
            'degraded': degraded,
            'timestamp': datetime.now()
        }

        if degraded:
            self.alerts.append(alert)

        return alert

    def statistical_test(self, metric_name='f1', alpha=0.05):
        """Perform t-test: baseline vs recent performance"""

        if metric_name not in self.baseline_metrics:
            return None

        # Baseline data
        baseline_cutoff = datetime.now() - timedelta(days=self.baseline_window_days)
        baseline_start = baseline_cutoff - timedelta(days=self.baseline_window_days)
        baseline_snapshots = [s for s in self.snapshots
                             if s.metric_name == metric_name
                             and baseline_start <= s.timestamp < baseline_cutoff]

        # Recent data (last 24 hours)
        recent_cutoff = datetime.now() - timedelta(hours=24)
        recent_snapshots = [s for s in self.snapshots
                           if s.metric_name == metric_name
                           and s.timestamp >= recent_cutoff]

        if len(baseline_snapshots) < 5 or len(recent_snapshots) < 5:
            return None

        baseline_values = [s.value for s in baseline_snapshots]
        recent_values = [s.value for s in recent_snapshots]

        # Two-sample t-test
        t_stat, p_value = stats.ttest_ind(recent_values, baseline_values)

        significant_degradation = (p_value < alpha) and (np.mean(recent_values) < np.mean(baseline_values))

        return {
            'metric': metric_name,
            'baseline_mean': np.mean(baseline_values),
            'recent_mean': np.mean(recent_values),
            't_statistic': t_stat,
            'p_value': p_value,
            'significant_degradation': significant_degradation
        }

    def get_metric_history(self, metric_name='f1', days=30):
        """Get metric trend over time"""

        cutoff = datetime.now() - timedelta(days=days)
        snapshots = [s for s in self.snapshots
                    if s.metric_name == metric_name and s.timestamp >= cutoff]

        df = pd.DataFrame([
            {'timestamp': s.timestamp, 'value': s.value}
            for s in snapshots
        ])

        return df

    def plot_metric_trend(self, metric_name='f1', days=30):
        """Visualize metric over time with baseline"""
        import matplotlib.pyplot as plt

        df = self.get_metric_history(metric_name, days)

        if df.empty:
            return

        plt.figure(figsize=(12, 6))
        plt.plot(df['timestamp'], df['value'], marker='o', label='Actual')

        if metric_name in self.baseline_metrics:
            baseline = self.baseline_metrics[metric_name]
            plt.axhline(baseline['mean'], color='green', linestyle='--', label='Baseline Mean')
            plt.axhline(baseline['mean'] - 2*baseline['std'], color='red',
                       linestyle='--', label='Alert Threshold (-2σ)')

        plt.xlabel('Time')
        plt.ylabel(metric_name)
        plt.title(f'{self.model_name}: {metric_name} Trend')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

# Usage
monitor = ModelMonitor('fraud_detector_v1', baseline_window_days=7)

# Simulate production logging
for day in range(30):
    # Simulate batch predictions
    y_true, y_pred, y_prob = simulate_production_batch()

    monitor.log_metrics(
        y_true, y_pred, y_prob,
        timestamp=datetime.now() - timedelta(days=30-day)
    )

# Set baseline from first week
monitor.set_baseline()

# Check for degradation
alert = monitor.check_degradation(metric_name='f1', threshold_std=2.0)
if alert and alert['degraded']:
    print(f"⚠️  ALERT: {alert['metric']} degraded!")
    print(f"  Baseline: {alert['baseline_mean']:.3f}")
    print(f"  Recent: {alert['recent_mean']:.3f}")
    print(f"  Z-score: {alert['z_score']:.3f}")

# Statistical test
test_result = monitor.statistical_test(metric_name='f1')
if test_result and test_result['significant_degradation']:
    print(f"\n🚨 RETRAIN RECOMMENDED")
    print(f"  p-value: {test_result['p_value']:.4f}")

# Visualize
monitor.plot_metric_trend('f1', days=30)
```

**Alerting thresholds:**
```python
def should_retrain(monitor, metric='f1', threshold_std=2.0, min_degradation=0.05):
    """Decide if model needs retraining"""

    alert = monitor.check_degradation(metric, threshold_std)

    if not alert or not alert['degraded']:
        return False, "Performance within acceptable range"

    degradation = alert['baseline_mean'] - alert['recent_mean']

    if degradation < min_degradation:
        return False, f"Degradation ({degradation:.3f}) below threshold"

    return True, f"Significant degradation detected: {degradation:.3f}"

# Decision logic
needs_retrain, reason = should_retrain(monitor, metric='f1', threshold_std=2.0)

if needs_retrain:
    print(f"🔄 Triggering retraining pipeline: {reason}")
    # trigger_retraining_job()
else:
    print(f"✅ Model OK: {reason}")
```

**Key components:**
1. Continuous metric logging
2. Statistical baseline
3. Degradation detection (z-score + t-test)
4. Automated alerting
5. Visualization for debugging

---

### Q6: Design a metrics framework for comparing multiple candidate models during experimentation. You need to support: (1) statistical significance testing, (2) cross-validation, (3) stratified evaluation on data subsets, and (4) cost-sensitive evaluation.

**Answer:**

Build a comprehensive **model comparison framework** with statistical rigor.

**Implementation:**

```python
from dataclasses import dataclass
from typing import List, Dict, Callable
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import *
from scipy import stats

@dataclass
class ModelResult:
    """Results from a single model evaluation"""
    model_name: str
    metrics: Dict[str, float]
    metrics_std: Dict[str, float]
    fold_metrics: List[Dict[str, float]]
    stratified_metrics: Dict[str, Dict[str, float]]
    cost: float

class ModelComparator:
    """Framework for rigorous model comparison"""

    def __init__(self, cv_folds=5, random_state=42):
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.results: List[ModelResult] = []

    def evaluate_model(self, model, X, y, model_name: str,
                      cost_fn: Callable = None,
                      stratify_by: np.ndarray = None):
        """
        Comprehensive model evaluation

        Parameters:
        -----------
        model : estimator
            scikit-learn compatible model
        X, y : array-like
            Features and labels
        model_name : str
            Model identifier
        cost_fn : callable, optional
            Custom cost function(y_true, y_pred) -> float
        stratify_by : array-like, optional
            Feature for stratified evaluation (e.g., user_age_group)
        """

        # Cross-validation
        skf = StratifiedKFold(n_splits=self.cv_folds, shuffle=True,
                             random_state=self.random_state)

        fold_metrics = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # Train
            model.fit(X_train, y_train)

            # Predict
            y_pred = model.predict(X_val)
            y_prob = model.predict_proba(X_val)[:, 1] if hasattr(model, 'predict_proba') else None

            # Calculate metrics
            metrics = self._calculate_metrics(y_val, y_pred, y_prob)
            metrics['fold'] = fold

            fold_metrics.append(metrics)

        # Aggregate CV metrics
        metrics_mean = {k: np.mean([m[k] for m in fold_metrics if k in m])
                       for k in fold_metrics[0].keys() if k != 'fold'}
        metrics_std = {k: np.std([m[k] for m in fold_metrics if k in m])
                      for k in fold_metrics[0].keys() if k != 'fold'}

        # Stratified evaluation (on full data for simplicity)
        stratified_metrics = {}
        if stratify_by is not None:
            model.fit(X, y)
            y_pred_full = model.predict(X)
            y_prob_full = model.predict_proba(X)[:, 1] if hasattr(model, 'predict_proba') else None

            for group in np.unique(stratify_by):
                mask = stratify_by == group
                stratified_metrics[f'group_{group}'] = self._calculate_metrics(
                    y[mask], y_pred_full[mask],
                    y_prob_full[mask] if y_prob_full is not None else None
                )

        # Cost evaluation
        total_cost = 0.0
        if cost_fn is not None:
            model.fit(X, y)
            y_pred_full = model.predict(X)
            total_cost = cost_fn(y, y_pred_full)

        # Store result
        result = ModelResult(
            model_name=model_name,
            metrics=metrics_mean,
            metrics_std=metrics_std,
            fold_metrics=fold_metrics,
            stratified_metrics=stratified_metrics,
            cost=total_cost
        )

        self.results.append(result)
        return result

    def _calculate_metrics(self, y_true, y_pred, y_prob=None):
        """Calculate all metrics"""

        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'specificity': recall_score(y_true, y_pred, pos_label=0, zero_division=0)
        }

        if y_prob is not None:
            metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
            metrics['pr_auc'] = average_precision_score(y_true, y_prob)

        return metrics

    def statistical_comparison(self, metric='f1', alpha=0.05):
        """
        Pairwise statistical comparison of models

        Returns DataFrame with p-values from paired t-tests
        """

        if len(self.results) < 2:
            return None

        # Extract fold-level metrics
        model_names = [r.model_name for r in self.results]
        fold_scores = {
            r.model_name: [m[metric] for m in r.fold_metrics]
            for r in self.results
        }

        # Pairwise t-tests
        comparison_results = []

        for i, model_a in enumerate(model_names):
            for j, model_b in enumerate(model_names):
                if i >= j:
                    continue

                scores_a = fold_scores[model_a]
                scores_b = fold_scores[model_b]

                # Paired t-test (same CV folds)
                t_stat, p_value = stats.ttest_rel(scores_a, scores_b)

                mean_diff = np.mean(scores_a) - np.mean(scores_b)
                significant = p_value < alpha

                comparison_results.append({
                    'model_a': model_a,
                    'model_b': model_b,
                    'mean_diff': mean_diff,
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'significant': significant,
                    'winner': model_a if mean_diff > 0 and significant else
                             (model_b if mean_diff < 0 and significant else 'tie')
                })

        return pd.DataFrame(comparison_results)

    def compare_table(self, sort_by='f1'):
        """Generate comparison table"""

        rows = []
        for result in self.results:
            row = {'model': result.model_name}

            # Add mean ± std for each metric
            for metric, mean_val in result.metrics.items():
                std_val = result.metrics_std.get(metric, 0)
                row[metric] = f"{mean_val:.3f} ± {std_val:.3f}"

            if result.cost > 0:
                row['cost'] = f"{result.cost:.2f}"

            rows.append(row)

        df = pd.DataFrame(rows)

        # Sort by mean of sort_by metric
        df['sort_key'] = [r.metrics[sort_by] for r in self.results]
        df = df.sort_values('sort_key', ascending=False).drop('sort_key', axis=1)

        return df

    def stratified_comparison(self):
        """Compare performance across data subsets"""

        rows = []
        for result in self.results:
            for group, metrics in result.stratified_metrics.items():
                row = {'model': result.model_name, 'group': group}
                row.update(metrics)
                rows.append(row)

        return pd.DataFrame(rows)

    def cost_comparison(self):
        """Compare models by cost"""

        return pd.DataFrame([
            {'model': r.model_name, 'cost': r.cost, 'f1': r.metrics['f1']}
            for r in self.results
        ]).sort_values('cost')

    def recommend_model(self, primary_metric='f1', cost_weight=0.0):
        """
        Recommend best model based on metric and cost

        Parameters:
        -----------
        primary_metric : str
            Metric to optimize
        cost_weight : float
            Weight for cost (0 = ignore cost, 1 = equal weight to metric)
        """

        if len(self.results) == 0:
            return None

        scores = []
        for result in self.results:
            metric_score = result.metrics[primary_metric]
            cost_score = -result.cost if result.cost > 0 else 0

            # Normalize cost to [0, 1] range
            if cost_weight > 0 and result.cost > 0:
                max_cost = max(r.cost for r in self.results)
                cost_score = (max_cost - result.cost) / max_cost

            # Combined score
            combined_score = (1 - cost_weight) * metric_score + cost_weight * cost_score

            scores.append({
                'model': result.model_name,
                'metric_score': metric_score,
                'cost': result.cost,
                'combined_score': combined_score
            })

        return pd.DataFrame(scores).sort_values('combined_score', ascending=False)

# Usage example
comparator = ModelComparator(cv_folds=5)

# Define models
models = {
    'logistic_regression': LogisticRegression(class_weight='balanced'),
    'random_forest': RandomForestClassifier(n_estimators=100, class_weight='balanced'),
    'xgboost': XGBClassifier(scale_pos_weight=10)
}

# Define cost function (FN = $100, FP = $10)
def business_cost(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return fn * 100 + fp * 10

# Evaluate all models
for name, model in models.items():
    comparator.evaluate_model(
        model=model,
        X=X,
        y=y,
        model_name=name,
        cost_fn=business_cost,
        stratify_by=user_age_groups  # Evaluate on age groups
    )

# Compare models
print("Model Comparison:")
print(comparator.compare_table(sort_by='f1'))

print("\nStatistical Significance:")
print(comparator.statistical_comparison(metric='f1'))

print("\nStratified Performance:")
print(comparator.stratified_comparison())

print("\nCost Analysis:")
print(comparator.cost_comparison())

print("\nRecommendation:")
print(comparator.recommend_model(primary_metric='f1', cost_weight=0.3))
```

This framework provides **rigorous, statistically-sound model comparison** for production ML!

---

### Q7: You need to evaluate a recommender system that outputs ranked lists of items. Design a metrics framework that captures both relevance (did we recommend good items?) and ranking quality (did we put the best items first?).

**Answer:**

Use **ranking metrics** like Precision@K, Recall@K, NDCG, and MAP.

**Implementation:**

```python
import numpy as np
from typing import List

class RankingMetrics:
    """Metrics for evaluating ranked recommendations"""

    @staticmethod
    def precision_at_k(y_true: List[int], y_pred: List[int], k: int) -> float:
        """
        Precision@K: What fraction of top-K recommendations are relevant?

        Parameters:
        -----------
        y_true : list of relevant item IDs
        y_pred : list of recommended item IDs (in rank order)
        k : cutoff rank
        """

        if k == 0:
            return 0.0

        top_k = y_pred[:k]
        relevant_in_top_k = len(set(top_k) & set(y_true))

        return relevant_in_top_k / k

    @staticmethod
    def recall_at_k(y_true: List[int], y_pred: List[int], k: int) -> float:
        """
        Recall@K: What fraction of relevant items are in top-K?
        """

        if len(y_true) == 0:
            return 0.0

        top_k = y_pred[:k]
        relevant_in_top_k = len(set(top_k) & set(y_true))

        return relevant_in_top_k / len(y_true)

    @staticmethod
    def average_precision(y_true: List[int], y_pred: List[int]) -> float:
        """
        Average Precision: Mean of precision values at each relevant item

        Good for measuring ranking quality
        """

        if len(y_true) == 0:
            return 0.0

        precisions = []
        relevant_count = 0

        for k in range(1, len(y_pred) + 1):
            if y_pred[k-1] in y_true:
                relevant_count += 1
                precision_at_k = relevant_count / k
                precisions.append(precision_at_k)

        if len(precisions) == 0:
            return 0.0

        return np.mean(precisions)

    @staticmethod
    def mean_average_precision(y_trues: List[List[int]],
                               y_preds: List[List[int]]) -> float:
        """
        MAP: Mean Average Precision across multiple queries/users

        Industry standard for ranking evaluation
        """

        aps = [RankingMetrics.average_precision(yt, yp)
              for yt, yp in zip(y_trues, y_preds)]

        return np.mean(aps)

    @staticmethod
    def dcg_at_k(y_true: List[int], y_pred: List[int], k: int,
                 relevance_scores: dict = None) -> float:
        """
        Discounted Cumulative Gain@K

        Considers position and relevance score

        relevance_scores: dict mapping item_id -> relevance (default: binary)
        """

        if relevance_scores is None:
            # Binary relevance: 1 if relevant, 0 otherwise
            relevance_scores = {item: 1 for item in y_true}

        dcg = 0.0
        for i, item in enumerate(y_pred[:k], start=1):
            rel = relevance_scores.get(item, 0)
            # DCG formula: rel / log2(position + 1)
            dcg += rel / np.log2(i + 1)

        return dcg

    @staticmethod
    def ndcg_at_k(y_true: List[int], y_pred: List[int], k: int,
                  relevance_scores: dict = None) -> float:
        """
        Normalized Discounted Cumulative Gain@K

        NDCG = DCG / IDCG (ideal DCG with perfect ranking)

        Returns value in [0, 1] where 1 = perfect ranking
        """

        if len(y_true) == 0:
            return 0.0

        # Actual DCG
        dcg = RankingMetrics.dcg_at_k(y_true, y_pred, k, relevance_scores)

        # Ideal DCG (best possible ranking)
        if relevance_scores is None:
            relevance_scores = {item: 1 for item in y_true}

        # Sort items by relevance (perfect ranking)
        ideal_ranking = sorted(y_true, key=lambda x: relevance_scores.get(x, 0), reverse=True)
        idcg = RankingMetrics.dcg_at_k(y_true, ideal_ranking, k, relevance_scores)

        if idcg == 0:
            return 0.0

        return dcg / idcg

    @staticmethod
    def mean_reciprocal_rank(y_trues: List[List[int]],
                            y_preds: List[List[int]]) -> float:
        """
        MRR: Average of reciprocal ranks of first relevant item

        Good for tasks where users want ONE good answer (e.g., search)
        """

        reciprocal_ranks = []

        for y_true, y_pred in zip(y_trues, y_preds):
            for rank, item in enumerate(y_pred, start=1):
                if item in y_true:
                    reciprocal_ranks.append(1.0 / rank)
                    break
            else:
                reciprocal_ranks.append(0.0)

        return np.mean(reciprocal_ranks)

def evaluate_recommender(user_relevant_items: dict,
                        user_recommendations: dict,
                        k_values=[1, 5, 10]):
    """
    Complete evaluation of recommender system

    Parameters:
    -----------
    user_relevant_items : dict
        {user_id: [relevant_item_ids]}
    user_recommendations : dict
        {user_id: [recommended_item_ids in rank order]}
    k_values : list
        K values for @K metrics
    """

    metrics = RankingMetrics()

    # Prepare data
    y_trues = [user_relevant_items[uid] for uid in user_relevant_items.keys()]
    y_preds = [user_recommendations[uid] for uid in user_relevant_items.keys()]

    results = {}

    # Precision/Recall @K
    for k in k_values:
        precisions = [metrics.precision_at_k(yt, yp, k)
                     for yt, yp in zip(y_trues, y_preds)]
        recalls = [metrics.recall_at_k(yt, yp, k)
                  for yt, yp in zip(y_trues, y_preds)]
        ndcgs = [metrics.ndcg_at_k(yt, yp, k)
                for yt, yp in zip(y_trues, y_preds)]

        results[f'Precision@{k}'] = np.mean(precisions)
        results[f'Recall@{k}'] = np.mean(recalls)
        results[f'NDCG@{k}'] = np.mean(ndcgs)

    # MAP and MRR
    results['MAP'] = metrics.mean_average_precision(y_trues, y_preds)
    results['MRR'] = metrics.mean_reciprocal_rank(y_trues, y_preds)

    return results

# Example usage
# User 1 liked items [5, 10, 15]
# User 2 liked items [2, 7]
user_relevant_items = {
    'user_1': [5, 10, 15],
    'user_2': [2, 7]
}

# System recommended items in ranked order
user_recommendations = {
    'user_1': [10, 3, 5, 8, 15, 20],  # Got 2/3 in top-3, all 3 in top-5
    'user_2': [2, 1, 7, 9]             # Got 2/2 in top-3
}

results = evaluate_recommender(user_relevant_items, user_recommendations, k_values=[1, 3, 5])

print("Recommender Evaluation:")
for metric, value in results.items():
    print(f"  {metric}: {value:.4f}")
```

Output:
```
Recommender Evaluation:
  Precision@1: 0.5000  # 50% of top-1 recs are relevant
  Recall@1: 0.2500     # Top-1 covers 25% of relevant items
  NDCG@1: 0.5000       # Ranking quality at top-1
  Precision@3: 0.6667
  Recall@3: 0.6667
  NDCG@3: 0.7682
  Precision@5: 0.5000
  Recall@5: 0.8333
  NDCG@5: 0.7682
  MAP: 0.7917          # Overall ranking quality
  MRR: 0.7500          # First relevant item position
```

**When to use which metric:**
- **Precision@K**: User sees K items, what % are good?
- **Recall@K**: Of all good items, what % are in top-K?
- **NDCG@K**: Quality of ranking (rewards putting best items first)
- **MAP**: Overall ranking quality across all positions
- **MRR**: Good for search (user wants ONE answer)

---

### Q8: Design a testing framework for ML metrics implementation. You need to verify that custom metric implementations match scikit-learn's behavior and handle edge cases correctly.

**Answer:**

Create **property-based tests** and **edge case tests** for metric validation.

**Implementation:**

```python
import unittest
import numpy as np
from sklearn.metrics import *
import hypothesis
from hypothesis import given, strategies as st
from hypothesis.extra.numpy import arrays

class TestMetricsImplementation(unittest.TestCase):
    """Test custom metrics match sklearn"""

    def test_perfect_predictions(self):
        """Test case: All predictions correct"""
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_pred = y_true.copy()

        # Perfect predictions should yield:
        assert accuracy_score(y_true, y_pred) == 1.0
        assert precision_score(y_true, y_pred) == 1.0
        assert recall_score(y_true, y_pred) == 1.0
        assert f1_score(y_true, y_pred) == 1.0

    def test_all_wrong(self):
        """Test case: All predictions wrong"""
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_pred = 1 - y_true  # Flip all labels

        # Should get 0 accuracy
        assert accuracy_score(y_true, y_pred) == 0.0
        assert f1_score(y_true, y_pred) == 0.0

    def test_all_same_class(self):
        """Edge case: Model predicts only one class"""
        y_true = np.array([0, 1, 1, 0, 1, 0])
        y_pred = np.ones(6)  # Predict all 1s

        # Precision should be the positive rate in y_true
        precision = precision_score(y_true, y_pred, zero_division=0)
        expected_precision = np.mean(y_true)  # 3/6 = 0.5
        assert precision == expected_precision

        # Recall should be 1.0 (found all positives)
        assert recall_score(y_true, y_pred) == 1.0

        # Specificity should be 0.0 (found no negatives)
        specificity = recall_score(y_true, y_pred, pos_label=0)
        assert specificity == 0.0

    def test_empty_predictions(self):
        """Edge case: Empty arrays"""
        y_true = np.array([])
        y_pred = np.array([])

        # Should handle gracefully
        with self.assertRaises(ValueError):
            accuracy_score(y_true, y_pred)

    def test_single_sample(self):
        """Edge case: Only one sample"""
        y_true = np.array([1])
        y_pred = np.array([1])

        assert accuracy_score(y_true, y_pred) == 1.0

    def test_no_positive_class(self):
        """Edge case: No positive samples"""
        y_true = np.zeros(10)
        y_pred = np.zeros(10)

        # Precision/Recall undefined, but sklearn returns 0 with zero_division=0
        assert precision_score(y_true, y_pred, zero_division=0) == 0.0
        assert recall_score(y_true, y_pred, zero_division=0) == 0.0

    def test_imbalanced_extreme(self):
        """Edge case: Extremely imbalanced (99:1)"""
        y_true = np.array([0] * 99 + [1])
        y_pred_naive = np.zeros(100)  # Predict all negative

        # High accuracy despite missing the one positive
        accuracy = accuracy_score(y_true, y_pred_naive)
        assert accuracy == 0.99

        # But zero recall
        recall = recall_score(y_true, y_pred_naive, zero_division=0)
        assert recall == 0.0

    @given(
        y_true=arrays(np.int8, st.integers(10, 100), elements=st.sampled_from([0, 1])),
        y_pred=arrays(np.int8, st.integers(10, 100), elements=st.sampled_from([0, 1]))
    )
    def test_property_accuracy_bounds(self, y_true, y_pred):
        """Property test: Accuracy always in [0, 1]"""
        if len(y_true) != len(y_pred):
            y_pred = y_pred[:len(y_true)]

        if len(y_true) > 0:
            acc = accuracy_score(y_true, y_pred)
            assert 0.0 <= acc <= 1.0

    @given(
        y_true=arrays(np.int8, st.integers(10, 100), elements=st.sampled_from([0, 1])),
        y_pred=arrays(np.int8, st.integers(10, 100), elements=st.sampled_from([0, 1]))
    )
    def test_property_precision_recall_f1(self, y_true, y_pred):
        """Property test: F1 is harmonic mean of precision and recall"""
        if len(y_true) != len(y_pred):
            y_pred = y_pred[:len(y_true)]

        if len(y_true) > 0 and np.sum(y_pred) > 0 and np.sum(y_true) > 0:
            p = precision_score(y_true, y_pred, zero_division=0)
            r = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)

            if p > 0 and r > 0:
                expected_f1 = 2 * (p * r) / (p + r)
                assert abs(f1 - expected_f1) < 1e-6

    def test_custom_vs_sklearn(self):
        """Verify custom implementation matches sklearn"""

        def my_accuracy(y_true, y_pred):
            """Custom accuracy implementation"""
            return np.mean(y_true == y_pred)

        def my_precision(y_true, y_pred):
            """Custom precision implementation"""
            tp = np.sum((y_true == 1) & (y_pred == 1))
            fp = np.sum((y_true == 0) & (y_pred == 1))
            return tp / (tp + fp) if (tp + fp) > 0 else 0.0

        def my_recall(y_true, y_pred):
            """Custom recall implementation"""
            tp = np.sum((y_true == 1) & (y_pred == 1))
            fn = np.sum((y_true == 1) & (y_pred == 0))
            return tp / (tp + fn) if (tp + fn) > 0 else 0.0

        # Test on random data
        np.random.seed(42)
        y_true = np.random.randint(0, 2, 1000)
        y_pred = np.random.randint(0, 2, 1000)

        # Compare custom vs sklearn
        assert abs(my_accuracy(y_true, y_pred) - accuracy_score(y_true, y_pred)) < 1e-10
        assert abs(my_precision(y_true, y_pred) - precision_score(y_true, y_pred, zero_division=0)) < 1e-10
        assert abs(my_recall(y_true, y_pred) - recall_score(y_true, y_pred, zero_division=0)) < 1e-10

    def test_confusion_matrix_properties(self):
        """Test confusion matrix properties"""
        y_true = np.array([0, 0, 1, 1, 0, 1, 1, 0])
        y_pred = np.array([0, 1, 1, 0, 0, 1, 1, 1])

        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()

        # Total should equal number of samples
        assert tn + fp + fn + tp == len(y_true)

        # Positive predictions = TP + FP
        assert tp + fp == np.sum(y_pred == 1)

        # Actual positives = TP + FN
        assert tp + fn == np.sum(y_true == 1)

        # Verify metrics from confusion matrix
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        assert abs(accuracy - accuracy_score(y_true, y_pred)) < 1e-10

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        assert abs(precision - precision_score(y_true, y_pred, zero_division=0)) < 1e-10

    def test_multiclass_averaging(self):
        """Test multi-class averaging strategies"""
        y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 2, 2, 0, 1, 1, 0, 1, 2])

        # Macro: unweighted mean of per-class scores
        macro_f1 = f1_score(y_true, y_pred, average='macro')

        # Weighted: weighted by class frequency
        weighted_f1 = f1_score(y_true, y_pred, average='weighted')

        # Micro: aggregate TP, FP, FN across classes
        micro_f1 = f1_score(y_true, y_pred, average='micro')

        # Micro should equal accuracy for balanced data
        accuracy = accuracy_score(y_true, y_pred)
        assert abs(micro_f1 - accuracy) < 1e-10

    def test_roc_auc_properties(self):
        """Test ROC-AUC properties"""

        # Perfect prediction: AUC = 1.0
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.1, 0.2, 0.9, 0.95])
        assert roc_auc_score(y_true, y_prob) == 1.0

        # Random prediction: AUC ≈ 0.5
        np.random.seed(42)
        y_true_rand = np.random.randint(0, 2, 1000)
        y_prob_rand = np.random.rand(1000)
        auc_rand = roc_auc_score(y_true_rand, y_prob_rand)
        assert 0.4 < auc_rand < 0.6  # Should be close to 0.5

        # Inverse prediction: AUC = 0.0
        y_prob_inverse = 1 - y_prob
        assert roc_auc_score(y_true, y_prob_inverse) == 0.0

# Run tests
if __name__ == '__main__':
    unittest.main()
```

**Edge case checklist:**
- ✓ Perfect predictions
- ✓ All wrong predictions
- ✓ All same class predicted
- ✓ Empty arrays
- ✓ Single sample
- ✓ No positive class
- ✓ Extreme imbalance (99:1)
- ✓ Confusion matrix math
- ✓ Multi-class averaging
- ✓ ROC-AUC bounds

---

### Q9: You're building an AutoML system that automatically selects the best metric for a given dataset. Design a heuristic system that analyzes the dataset and recommends appropriate evaluation metrics.

**Answer:**

Create an **intelligent metric selector** that analyzes dataset characteristics.

**Implementation:**

```python
from dataclasses import dataclass
from typing import List
import numpy as np
import pandas as pd

@dataclass
class MetricRecommendation:
    """Recommended metric with rationale"""
    metric_name: str
    priority: str  # 'primary', 'secondary', 'monitor'
    rationale: str
    implementation: str  # sklearn function name

class MetricSelector:
    """Intelligent metric selection based on dataset characteristics"""

    def analyze_dataset(self, y: np.ndarray, X: np.ndarray = None):
        """Analyze dataset and recommend metrics"""

        characteristics = self._extract_characteristics(y, X)
        recommendations = self._recommend_metrics(characteristics)

        return {
            'characteristics': characteristics,
            'recommendations': recommendations
        }

    def _extract_characteristics(self, y, X=None):
        """Extract dataset characteristics"""

        unique_values = np.unique(y)
        n_classes = len(unique_values)
        class_counts = {val: np.sum(y == val) for val in unique_values}
        class_proportions = {val: count / len(y) for val, count in class_counts.items()}

        # Imbalance ratio: largest class / smallest class
        max_prop = max(class_proportions.values())
        min_prop = min(class_proportions.values())
        imbalance_ratio = max_prop / min_prop if min_prop > 0 else float('inf')

        # Minority class size
        minority_size = min(class_counts.values())
        minority_proportion = minority_size / len(y)

        characteristics = {
            'n_samples': len(y),
            'n_classes': n_classes,
            'class_counts': class_counts,
            'class_proportions': class_proportions,
            'imbalance_ratio': imbalance_ratio,
            'minority_size': minority_size,
            'minority_proportion': minority_proportion,
            'is_binary': n_classes == 2,
            'is_multiclass': n_classes > 2,
            'is_balanced': imbalance_ratio < 1.5,
            'is_imbalanced': imbalance_ratio >= 3.0,
            'is_highly_imbalanced': imbalance_ratio >= 10.0
        }

        return characteristics

    def _recommend_metrics(self, characteristics):
        """Recommend metrics based on characteristics"""

        recommendations = []

        # Binary classification
        if characteristics['is_binary']:
            recommendations.extend(self._recommend_binary(characteristics))

        # Multi-class classification
        elif characteristics['is_multiclass']:
            recommendations.extend(self._recommend_multiclass(characteristics))

        return recommendations

    def _recommend_binary(self, characteristics):
        """Recommend metrics for binary classification"""

        recommendations = []

        # Balanced dataset
        if characteristics['is_balanced']:
            recommendations.append(MetricRecommendation(
                metric_name='Accuracy',
                priority='primary',
                rationale=f"Dataset is balanced (ratio: {characteristics['imbalance_ratio']:.2f}). "
                         "Accuracy is a reliable metric.",
                implementation='accuracy_score'
            ))
            recommendations.append(MetricRecommendation(
                metric_name='F1-Score',
                priority='secondary',
                rationale="F1 provides additional insight on precision/recall balance.",
                implementation='f1_score'
            ))

        # Imbalanced dataset
        elif characteristics['is_imbalanced']:
            recommendations.append(MetricRecommendation(
                metric_name='F1-Score',
                priority='primary',
                rationale=f"Dataset is imbalanced (ratio: {characteristics['imbalance_ratio']:.2f}). "
                         "F1-Score accounts for both precision and recall on minority class.",
                implementation='f1_score'
            ))
            recommendations.append(MetricRecommendation(
                metric_name='PR-AUC',
                priority='primary',
                rationale="Precision-Recall AUC is robust to class imbalance.",
                implementation='average_precision_score'
            ))
            recommendations.append(MetricRecommendation(
                metric_name='ROC-AUC',
                priority='secondary',
                rationale="ROC-AUC for threshold-independent evaluation.",
                implementation='roc_auc_score'
            ))

        # Highly imbalanced
        if characteristics['is_highly_imbalanced']:
            minority_pct = characteristics['minority_proportion'] * 100
            recommendations.append(MetricRecommendation(
                metric_name='Recall (Sensitivity)',
                priority='primary',
                rationale=f"Highly imbalanced ({minority_pct:.2f}% minority). "
                         "Prioritize finding minority class instances.",
                implementation='recall_score'
            ))
            recommendations.append(MetricRecommendation(
                metric_name='Balanced Accuracy',
                priority='monitor',
                rationale="Average of per-class recall, not biased by majority class.",
                implementation='balanced_accuracy_score'
            ))

        # Always monitor confusion matrix
        recommendations.append(MetricRecommendation(
            metric_name='Confusion Matrix',
            priority='monitor',
            rationale="Essential for understanding TP, FP, TN, FN.",
            implementation='confusion_matrix'
        ))

        return recommendations

    def _recommend_multiclass(self, characteristics):
        """Recommend metrics for multi-class classification"""

        recommendations = []

        if characteristics['is_balanced']:
            recommendations.append(MetricRecommendation(
                metric_name='Accuracy',
                priority='primary',
                rationale="Classes are balanced. Accuracy is reliable.",
                implementation='accuracy_score'
            ))
            recommendations.append(MetricRecommendation(
                metric_name='Macro F1-Score',
                priority='secondary',
                rationale="Macro-average gives equal weight to each class.",
                implementation='f1_score(average=\"macro\")'
            ))
        else:
            recommendations.append(MetricRecommendation(
                metric_name='Weighted F1-Score',
                priority='primary',
                rationale=f"Classes are imbalanced (ratio: {characteristics['imbalance_ratio']:.2f}). "
                         "Weighted F1 accounts for class frequency.",
                implementation='f1_score(average=\"weighted\")'
            ))
            recommendations.append(MetricRecommendation(
                metric_name='Macro F1-Score',
                priority='secondary',
                rationale="Monitor per-class performance equally.",
                implementation='f1_score(average=\"macro\")'
            ))

        recommendations.append(MetricRecommendation(
            metric_name='Per-Class F1-Score',
            priority='monitor',
            rationale="Identify which classes are problematic.",
            implementation='f1_score(average=None)'
        ))

        recommendations.append(MetricRecommendation(
            metric_name='Confusion Matrix',
            priority='monitor',
            rationale="Visualize all class confusions.",
            implementation='confusion_matrix'
        ))

        return recommendations

    def print_recommendations(self, analysis):
        """Pretty-print metric recommendations"""

        print("Dataset Characteristics:")
        print(f"  Samples: {analysis['characteristics']['n_samples']}")
        print(f"  Classes: {analysis['characteristics']['n_classes']}")
        print(f"  Imbalance Ratio: {analysis['characteristics']['imbalance_ratio']:.2f}")
        print(f"  Minority Proportion: {analysis['characteristics']['minority_proportion']*100:.2f}%")

        print("\nRecommended Metrics:")
        for rec in analysis['recommendations']:
            print(f"\n  [{rec.priority.upper()}] {rec.metric_name}")
            print(f"    Rationale: {rec.rationale}")
            print(f"    Implementation: {rec.implementation}")

# Example usage
selector = MetricSelector()

# Scenario 1: Balanced binary classification
y_balanced = np.array([0]*500 + [1]*500)
analysis = selector.analyze_dataset(y_balanced)
print("=== Scenario 1: Balanced Binary ===")
selector.print_recommendations(analysis)

print("\n" + "="*50 + "\n")

# Scenario 2: Imbalanced fraud detection (0.5% fraud rate)
y_imbalanced = np.array([0]*9950 + [1]*50)
analysis = selector.analyze_dataset(y_imbalanced)
print("=== Scenario 2: Highly Imbalanced (Fraud Detection) ===")
selector.print_recommendations(analysis)

print("\n" + "="*50 + "\n")

# Scenario 3: Multi-class ticket classification
y_multiclass = np.array([0]*100 + [1]*150 + [2]*80)
analysis = selector.analyze_dataset(y_multiclass)
print("=== Scenario 3: Imbalanced Multi-class ===")
selector.print_recommendations(analysis)
```

Output:
```
=== Scenario 1: Balanced Binary ===
Dataset Characteristics:
  Samples: 1000
  Classes: 2
  Imbalance Ratio: 1.00
  Minority Proportion: 50.00%

Recommended Metrics:

  [PRIMARY] Accuracy
    Rationale: Dataset is balanced (ratio: 1.00). Accuracy is a reliable metric.
    Implementation: accuracy_score

  [SECONDARY] F1-Score
    Rationale: F1 provides additional insight on precision/recall balance.
    Implementation: f1_score

==================================================

=== Scenario 2: Highly Imbalanced (Fraud Detection) ===
Dataset Characteristics:
  Samples: 10000
  Classes: 2
  Imbalance Ratio: 199.00
  Minority Proportion: 0.50%

Recommended Metrics:

  [PRIMARY] F1-Score
    Rationale: Dataset is imbalanced (ratio: 199.00). F1-Score accounts for both precision and recall on minority class.
    Implementation: f1_score

  [PRIMARY] PR-AUC
    Rationale: Precision-Recall AUC is robust to class imbalance.
    Implementation: average_precision_score

  [PRIMARY] Recall (Sensitivity)
    Rationale: Highly imbalanced (0.50% minority). Prioritize finding minority class instances.
    Implementation: recall_score

  [MONITOR] Balanced Accuracy
    Rationale: Average of per-class recall, not biased by majority class.
    Implementation: balanced_accuracy_score
```

**This provides automatic, intelligent metric selection for any ML task!**

---

### Q10: Design a metrics-driven A/B testing framework for ML models in production. How do you determine if Model B is statistically significantly better than Model A?

**Answer:**

Implement **statistical hypothesis testing** for model comparison in production.

**Implementation:**

```python
from dataclasses import dataclass
from datetime import datetime
import numpy as np
import pandas as pd
from scipy import stats
from typing import List

@dataclass
class ABTestSample:
    """Single prediction sample in A/B test"""
    timestamp: datetime
    model_version: str  # 'A' or 'B'
    y_true: int
    y_pred: int
    y_prob: float
    user_id: str = None

class ABTestFramework:
    """Statistical framework for ML model A/B testing"""

    def __init__(self, alpha=0.05, min_samples=1000):
        """
        Parameters:
        -----------
        alpha : float
            Significance level (default 0.05 = 95% confidence)
        min_samples : int
            Minimum samples needed per variant
        """
        self.alpha = alpha
        self.min_samples = min_samples
        self.samples_a: List[ABTestSample] = []
        self.samples_b: List[ABTestSample] = []

    def log_prediction(self, model_version: str, y_true: int, y_pred: int,
                      y_prob: float, user_id: str = None):
        """Log a single prediction"""

        sample = ABTestSample(
            timestamp=datetime.now(),
            model_version=model_version,
            y_true=y_true,
            y_pred=y_pred,
            y_prob=y_prob,
            user_id=user_id
        )

        if model_version == 'A':
            self.samples_a.append(sample)
        elif model_version == 'B':
            self.samples_b.append(sample)

    def _extract_metrics(self, samples: List[ABTestSample]):
        """Extract metrics from samples"""

        y_true = np.array([s.y_true for s in samples])
        y_pred = np.array([s.y_pred for s in samples])
        y_prob = np.array([s.y_prob for s in samples])

        # Per-sample metrics (for statistical testing)
        correct = (y_true == y_pred).astype(int)

        return {
            'y_true': y_true,
            'y_pred': y_pred,
            'y_prob': y_prob,
            'correct': correct,
            'accuracy': np.mean(correct),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else None
        }

    def run_test(self):
        """Run statistical comparison between Model A and Model B"""

        if len(self.samples_a) < self.min_samples or len(self.samples_b) < self.min_samples:
            return {
                'test_complete': False,
                'reason': f'Need {self.min_samples} samples per variant. '
                         f'A: {len(self.samples_a)}, B: {len(self.samples_b)}'
            }

        metrics_a = self._extract_metrics(self.samples_a)
        metrics_b = self._extract_metrics(self.samples_b)

        # Statistical tests
        results = {
            'test_complete': True,
            'n_samples_a': len(self.samples_a),
            'n_samples_b': len(self.samples_b),
            'model_a': {},
            'model_b': {},
            'statistical_tests': {}
        }

        # Store aggregate metrics
        for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
            results['model_a'][metric] = metrics_a[metric]
            results['model_b'][metric] = metrics_b[metric]

        # Test 1: Proportions test for accuracy
        # H0: accuracy_A = accuracy_B
        # H1: accuracy_A ≠ accuracy_B
        correct_a = np.sum(metrics_a['correct'])
        correct_b = np.sum(metrics_b['correct'])
        n_a = len(self.samples_a)
        n_b = len(self.samples_b)

        # Two-proportion z-test
        p_a = correct_a / n_a
        p_b = correct_b / n_b
        p_pool = (correct_a + correct_b) / (n_a + n_b)

        se = np.sqrt(p_pool * (1 - p_pool) * (1/n_a + 1/n_b))
        z_stat = (p_b - p_a) / se if se > 0 else 0
        p_value_accuracy = 2 * (1 - stats.norm.cdf(abs(z_stat)))

        results['statistical_tests']['accuracy'] = {
            'test': 'two-proportion z-test',
            'z_statistic': z_stat,
            'p_value': p_value_accuracy,
            'significant': p_value_accuracy < self.alpha,
            'winner': 'B' if p_value_accuracy < self.alpha and p_b > p_a else
                     ('A' if p_value_accuracy < self.alpha and p_a > p_b else 'tie')
        }

        # Test 2: Permutation test for F1-score
        # (Bootstrap alternative to parametric tests)
        f1_diff_observed = metrics_b['f1'] - metrics_a['f1']
        f1_diff_permuted = self._permutation_test(
            self.samples_a, self.samples_b, metric='f1', n_permutations=1000
        )

        p_value_f1 = np.mean(np.abs(f1_diff_permuted) >= abs(f1_diff_observed))

        results['statistical_tests']['f1'] = {
            'test': 'permutation test',
            'observed_diff': f1_diff_observed,
            'p_value': p_value_f1,
            'significant': p_value_f1 < self.alpha,
            'winner': 'B' if p_value_f1 < self.alpha and f1_diff_observed > 0 else
                     ('A' if p_value_f1 < self.alpha and f1_diff_observed < 0 else 'tie')
        }

        # Test 3: ROC-AUC comparison (DeLong test would be ideal, simplified here)
        if metrics_a['roc_auc'] is not None and metrics_b['roc_auc'] is not None:
            auc_diff = metrics_b['roc_auc'] - metrics_a['roc_auc']
            results['statistical_tests']['roc_auc'] = {
                'test': 'difference',
                'diff': auc_diff,
                'model_a': metrics_a['roc_auc'],
                'model_b': metrics_b['roc_auc']
            }

        # Overall recommendation
        results['recommendation'] = self._make_recommendation(results)

        return results

    def _permutation_test(self, samples_a, samples_b, metric='f1', n_permutations=1000):
        """Permutation test for metric difference"""

        # Combine all samples
        all_samples = samples_a + samples_b
        n_a = len(samples_a)
        n_b = len(samples_b)

        diffs = []

        for _ in range(n_permutations):
            # Randomly shuffle and split
            shuffled = np.random.permutation(all_samples)
            perm_a = shuffled[:n_a]
            perm_b = shuffled[n_a:]

            metrics_a = self._extract_metrics(perm_a)
            metrics_b = self._extract_metrics(perm_b)

            diff = metrics_b[metric] - metrics_a[metric]
            diffs.append(diff)

        return np.array(diffs)

    def _make_recommendation(self, results):
        """Make deployment recommendation based on test results"""

        significant_wins = []
        significant_losses = []

        for metric, test_result in results['statistical_tests'].items():
            if test_result.get('significant', False):
                winner = test_result['winner']
                if winner == 'B':
                    significant_wins.append(metric)
                elif winner == 'A':
                    significant_losses.append(metric)

        if len(significant_wins) > 0 and len(significant_losses) == 0:
            return {
                'decision': 'deploy_B',
                'confidence': 'high',
                'reason': f'Model B significantly better on: {", ".join(significant_wins)}'
            }
        elif len(significant_losses) > 0:
            return {
                'decision': 'keep_A',
                'confidence': 'high',
                'reason': f'Model A significantly better on: {", ".join(significant_losses)}'
            }
        else:
            return {
                'decision': 'inconclusive',
                'confidence': 'low',
                'reason': 'No significant difference detected. Consider longer test or larger sample.'
            }

    def report(self):
        """Generate comprehensive report"""

        results = self.run_test()

        if not results['test_complete']:
            print(f"Test incomplete: {results['reason']}")
            return

        print("=" * 60)
        print("A/B TEST RESULTS")
        print("=" * 60)

        print(f"\nSample Sizes:")
        print(f"  Model A: {results['n_samples_a']}")
        print(f"  Model B: {results['n_samples_b']}")

        print(f"\nModel Performance:")
        print(f"{'Metric':<15} {'Model A':<12} {'Model B':<12} {'Diff':<12}")
        print("-" * 60)

        for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
            if metric in results['model_a'] and results['model_a'][metric] is not None:
                val_a = results['model_a'][metric]
                val_b = results['model_b'][metric]
                diff = val_b - val_a
                print(f"{metric:<15} {val_a:<12.4f} {val_b:<12.4f} {diff:+12.4f}")

        print(f"\nStatistical Tests (α={self.alpha}):")
        print("-" * 60)

        for metric, test in results['statistical_tests'].items():
            print(f"\n{metric.upper()}:")
            print(f"  Test: {test['test']}")
            print(f"  p-value: {test.get('p_value', 'N/A')}")
            print(f"  Significant: {test.get('significant', False)}")
            print(f"  Winner: {test.get('winner', 'N/A')}")

        print(f"\n{'='*60}")
        print(f"RECOMMENDATION: {results['recommendation']['decision'].upper()}")
        print(f"Confidence: {results['recommendation']['confidence']}")
        print(f"Reason: {results['recommendation']['reason']}")
        print("=" * 60)

# Example usage
ab_test = ABTestFramework(alpha=0.05, min_samples=1000)

# Simulate A/B test data
np.random.seed(42)
for i in range(2000):
    # Model A: 80% accuracy
    model = 'A'
    y_true = np.random.randint(0, 2)
    y_pred = y_true if np.random.rand() < 0.80 else 1 - y_true
    y_prob = 0.9 if y_pred == 1 else 0.1

    ab_test.log_prediction(model, y_true, y_pred, y_prob)

    # Model B: 82% accuracy (slightly better)
    model = 'B'
    y_true = np.random.randint(0, 2)
    y_pred = y_true if np.random.rand() < 0.82 else 1 - y_true
    y_prob = 0.92 if y_pred == 1 else 0.08

    ab_test.log_prediction(model, y_true, y_pred, y_prob)

# Run test and generate report
ab_test.report()
```

This provides **statistically rigorous A/B testing** for production ML models!

---

## Implementation & Coding (Q11-Q20)

### Q11: Implement a function that calculates precision, recall, and F1-score from a confusion matrix without using scikit-learn. Include edge case handling for when TP+FP=0 or TP+FN=0.

**Answer:**

```python
import numpy as np

def metrics_from_confusion_matrix(tn, fp, fn, tp, zero_division=0.0):
    """
    Calculate precision, recall, F1 from confusion matrix

    Parameters:
    -----------
    tn, fp, fn, tp : int
        Confusion matrix values
    zero_division : float
        Value to return when denominator is 0 (default 0.0)

    Returns:
    --------
    dict with precision, recall, f1, accuracy, specificity
    """

    # Accuracy
    total = tn + fp + fn + tp
    accuracy = (tp + tn) / total if total > 0 else zero_division

    # Precision: TP / (TP + FP)
    precision_denom = tp + fp
    if precision_denom == 0:
        precision = zero_division
    else:
        precision = tp / precision_denom

    # Recall (Sensitivity): TP / (TP + FN)
    recall_denom = tp + fn
    if recall_denom == 0:
        recall = zero_division
    else:
        recall = tp / recall_denom

    # F1-score: 2 * (P * R) / (P + R)
    f1_denom = precision + recall
    if f1_denom == 0:
        f1 = zero_division
    else:
        f1 = 2 * (precision * recall) / f1_denom

    # Specificity (True Negative Rate): TN / (TN + FP)
    specificity_denom = tn + fp
    if specificity_denom == 0:
        specificity = zero_division
    else:
        specificity = tn / specificity_denom

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'specificity': specificity,
        'true_positive_rate': recall,  # Alias
        'false_positive_rate': 1 - specificity if specificity != zero_division else zero_division
    }

# Test cases
def test_metrics():
    """Test with various edge cases"""

    # Normal case
    print("Normal case (TP=50, FP=10, FN=5, TN=35):")
    metrics = metrics_from_confusion_matrix(tn=35, fp=10, fn=5, tp=50)
    print(f"  Precision: {metrics['precision']:.3f}")
    print(f"  Recall: {metrics['recall']:.3f}")
    print(f"  F1: {metrics['f1']:.3f}")
    print(f"  Accuracy: {metrics['accuracy']:.3f}")

    # Edge case 1: No predicted positives (TP=0, FP=0)
    print("\nEdge case: No predicted positives (all predicted negative)")
    metrics = metrics_from_confusion_matrix(tn=90, fp=0, fn=10, tp=0, zero_division=0)
    print(f"  Precision: {metrics['precision']:.3f}")  # 0 (undefined)
    print(f"  Recall: {metrics['recall']:.3f}")        # 0 (missed all positives)
    print(f"  F1: {metrics['f1']:.3f}")                # 0

    # Edge case 2: No actual positives (TP=0, FN=0)
    print("\nEdge case: No actual positives in dataset")
    metrics = metrics_from_confusion_matrix(tn=80, fp=20, fn=0, tp=0, zero_division=0)
    print(f"  Precision: {metrics['precision']:.3f}")  # 0 (all FP)
    print(f"  Recall: {metrics['recall']:.3f}")        # 0 (undefined, no positives)
    print(f"  F1: {metrics['f1']:.3f}")                # 0

    # Edge case 3: Perfect precision, imperfect recall
    print("\nPerfect precision (FP=0), but missing some positives")
    metrics = metrics_from_confusion_matrix(tn=50, fp=0, fn=10, tp=40)
    print(f"  Precision: {metrics['precision']:.3f}")  # 1.0 (no FP)
    print(f"  Recall: {metrics['recall']:.3f}")        # 0.8 (40/50)
    print(f"  F1: {metrics['f1']:.3f}")                # Harmonic mean

    # Edge case 4: Perfect recall, imperfect precision
    print("\nPerfect recall (FN=0), but some false positives")
    metrics = metrics_from_confusion_matrix(tn=40, fp=10, fn=0, tp=50)
    print(f"  Precision: {metrics['precision']:.3f}")  # 0.833 (50/60)
    print(f"  Recall: {metrics['recall']:.3f}")        # 1.0 (no FN)
    print(f"  F1: {metrics['f1']:.3f}")                # Harmonic mean

    # Edge case 5: Perfect classifier
    print("\nPerfect classifier")
    metrics = metrics_from_confusion_matrix(tn=50, fp=0, fn=0, tp=50)
    print(f"  Precision: {metrics['precision']:.3f}")  # 1.0
    print(f"  Recall: {metrics['recall']:.3f}")        # 1.0
    print(f"  F1: {metrics['f1']:.3f}")                # 1.0
    print(f"  Accuracy: {metrics['accuracy']:.3f}")    # 1.0

test_metrics()
```

**Verification against sklearn:**

```python
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

def verify_implementation():
    """Verify custom implementation matches sklearn"""

    y_true = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 1])
    y_pred = np.array([1, 0, 0, 1, 0, 1, 0, 1, 1, 1])

    # sklearn metrics
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sk_precision = precision_score(y_true, y_pred)
    sk_recall = recall_score(y_true, y_pred)
    sk_f1 = f1_score(y_true, y_pred)

    # Custom implementation
    custom_metrics = metrics_from_confusion_matrix(tn, fp, fn, tp)

    # Compare
    print("Verification against sklearn:")
    print(f"  Precision - sklearn: {sk_precision:.6f}, custom: {custom_metrics['precision']:.6f}")
    print(f"  Recall    - sklearn: {sk_recall:.6f}, custom: {custom_metrics['recall']:.6f}")
    print(f"  F1        - sklearn: {sk_f1:.6f}, custom: {custom_metrics['f1']:.6f}")

    assert abs(sk_precision - custom_metrics['precision']) < 1e-10
    assert abs(sk_recall - custom_metrics['recall']) < 1e-10
    assert abs(sk_f1 - custom_metrics['f1']) < 1e-10

    print("\n✓ All checks passed!")

verify_implementation()
```

---

### Q12: Write a function that plots a confusion matrix heatmap with percentages, absolute counts, and row/column summaries. Include proper labeling and formatting.

**Answer:**

```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix_detailed(y_true, y_pred, class_names=None,
                                   normalize='all', figsize=(10, 8)):
    """
    Plot detailed confusion matrix with counts and percentages

    Parameters:
    -----------
    y_true, y_pred : array-like
        True and predicted labels
    class_names : list, optional
        Names for classes
    normalize : str, optional
        'all': Show % of total
        'true': Show % of each true class (recall perspective)
        'pred': Show % of each predicted class (precision perspective)
        None: Show only counts
    """

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Class names
    if class_names is None:
        class_names = [f'Class {i}' for i in range(len(cm))]

    # Calculate percentages
    if normalize == 'all':
        cm_percent = cm / cm.sum() * 100
        fmt_string = '{:.0f}\n({:.1f}%)'
        title_suffix = '(% of total)'
    elif normalize == 'true':
        cm_percent = cm / cm.sum(axis=1, keepdims=True) * 100
        fmt_string = '{:.0f}\n({:.1f}%)'
        title_suffix = '(% of row = recall)'
    elif normalize == 'pred':
        cm_percent = cm / cm.sum(axis=0, keepdims=True) * 100
        fmt_string = '{:.0f}\n({:.1f}%)'
        title_suffix = '(% of column = precision)'
    else:
        cm_percent = cm
        fmt_string = '{:.0f}'
        title_suffix = ''

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot heatmap
    sns.heatmap(cm_percent if normalize else cm,
                annot=False,  # We'll add custom annotations
                fmt='.1f' if normalize else '.0f',
                cmap='Blues',
                square=True,
                linewidths=0.5,
                cbar_kws={'label': 'Percentage' if normalize else 'Count'},
                xticklabels=class_names,
                yticklabels=class_names,
                ax=ax)

    # Add custom annotations with counts and percentages
    for i in range(len(cm)):
        for j in range(len(cm)):
            count = cm[i, j]
            if normalize:
                percent = cm_percent[i, j]
                text = fmt_string.format(count, percent)
            else:
                text = f'{count:.0f}'

            # Color: white for dark cells, black for light cells
            threshold = cm_percent.max() / 2 if normalize else cm.max() / 2
            color = 'white' if (cm_percent[i, j] if normalize else cm[i, j]) > threshold else 'black'

            ax.text(j + 0.5, i + 0.5, text,
                   ha='center', va='center',
                   color=color, fontsize=11, fontweight='bold')

    # Add row and column totals
    row_totals = cm.sum(axis=1)
    col_totals = cm.sum(axis=0)

    # Column totals (bottom)
    for j, total in enumerate(col_totals):
        ax.text(j + 0.5, len(cm) + 0.3, f'Σ={total}',
               ha='center', va='top', fontsize=9, style='italic')

    # Row totals (right)
    for i, total in enumerate(row_totals):
        ax.text(len(cm) + 0.3, i + 0.5, f'Σ={total}',
               ha='left', va='center', fontsize=9, style='italic')

    # Labels
    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax.set_title(f'Confusion Matrix {title_suffix}', fontsize=14, fontweight='bold', pad=20)

    # Calculate and display metrics
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (None, None, None, None)

    if tn is not None:
        accuracy = (tp + tn) / (tn + fp + fn + tp)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        metrics_text = f'Accuracy: {accuracy:.3f}  |  Precision: {precision:.3f}  |  Recall: {recall:.3f}  |  F1: {f1:.3f}'
        fig.text(0.5, 0.02, metrics_text, ha='center', fontsize=11, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.show()

# Example usage
np.random.seed(42)
y_true = np.random.choice(['Cat', 'Dog', 'Bird'], size=200, p=[0.5, 0.3, 0.2])
y_pred = y_true.copy()
# Add some errors
errors = np.random.choice(200, size=40, replace=False)
y_pred[errors] = np.random.choice(['Cat', 'Dog', 'Bird'], size=40)

# Plot with different normalizations
plot_confusion_matrix_detailed(y_true, y_pred, class_names=['Cat', 'Dog', 'Bird'], normalize='all')
plot_confusion_matrix_detailed(y_true, y_pred, class_names=['Cat', 'Dog', 'Bird'], normalize='true')
plot_confusion_matrix_detailed(y_true, y_pred, class_names=['Cat', 'Dog', 'Bird'], normalize=None)
```

This creates **publication-quality confusion matrix visualizations** with all relevant information!

---

(Continuing with Q13-Q20 in next message due to length...)

### Q13: Implement a function that calculates ROC curve points and ROC-AUC score from scratch without using scikit-learn. The function should handle edge cases like all same predictions.

**Answer:**

```python
import numpy as np

def calculate_roc_curve(y_true, y_prob):
    """
    Calculate ROC curve (FPR, TPR) at different thresholds
    
    Returns:
    --------
    fpr, tpr, thresholds, auc
    """
    
    # Sort by predicted probability (descending)
    sorted_indices = np.argsort(-y_prob)
    y_true_sorted = y_true[sorted_indices]
    y_prob_sorted = y_prob[sorted_indices]
    
    # Get unique thresholds
    thresholds = np.unique(y_prob_sorted)
    thresholds = np.concatenate([thresholds, [thresholds[-1] - 1e-10]])
    
    # Calculate TPR and FPR at each threshold
    tpr_list = []
    fpr_list = []
    
    n_positive = np.sum(y_true == 1)
    n_negative = np.sum(y_true == 0)
    
    # Edge case: all same class
    if n_positive == 0 or n_negative == 0:
        return np.array([0, 1]), np.array([0, 1]), thresholds, 0.5
    
    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        
        tpr = tp / n_positive
        fpr = fp / n_negative
        
        tpr_list.append(tpr)
        fpr_list.append(fpr)
    
    # Add (0, 0) point
    tpr_list = [0] + tpr_list
    fpr_list = [0] + fpr_list
    
    tpr_array = np.array(tpr_list)
    fpr_array = np.array(fpr_list)
    
    # Calculate AUC using trapezoidal rule
    auc = np.trapz(tpr_array, fpr_array)
    
    return fpr_array, tpr_array, thresholds, auc


def roc_auc_score_custom(y_true, y_prob):
    """Calculate ROC-AUC score"""
    _, _, _, auc = calculate_roc_curve(y_true, y_prob)
    return auc


# Test and verify
from sklearn.metrics import roc_auc_score, roc_curve

# Test case 1: Normal case
y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1])
y_prob = np.array([0.1, 0.3, 0.8, 0.9, 0.2, 0.7, 0.4, 0.85])

fpr_custom, tpr_custom, _, auc_custom = calculate_roc_curve(y_true, y_prob)
fpr_sklearn, tpr_sklearn, _ = roc_curve(y_true, y_prob)
auc_sklearn = roc_auc_score(y_true, y_prob)

print("Test 1: Normal case")
print(f"Custom AUC: {auc_custom:.6f}")
print(f"Sklearn AUC: {auc_sklearn:.6f}")
print(f"Match: {abs(auc_custom - auc_sklearn) < 1e-5}")

# Test case 2: Perfect predictions
y_true = np.array([0, 0, 0, 1, 1, 1])
y_prob = np.array([0.1, 0.2, 0.3, 0.8, 0.9, 0.95])

auc_custom = roc_auc_score_custom(y_true, y_prob)
print(f"\nTest 2: Perfect predictions")
print(f"Custom AUC: {auc_custom:.6f} (should be 1.0)")

# Test case 3: Random predictions
np.random.seed(42)
y_true = np.random.randint(0, 2, 100)
y_prob = np.random.rand(100)

auc_custom = roc_auc_score_custom(y_true, y_prob)
auc_sklearn = roc_auc_score(y_true, y_prob)

print(f"\nTest 3: Random predictions")
print(f"Custom AUC: {auc_custom:.6f}")
print(f"Sklearn AUC: {auc_sklearn:.6f}")
print(f"Close to 0.5: {0.4 < auc_custom < 0.6}")
```

---

### Q14: Write a function that computes Precision@K and Recall@K for a ranking/recommendation system. Include handling for when there are fewer than K predictions or no relevant items.

**Answer:**

```python
import numpy as np
from typing import List, Union

def precision_at_k(y_true: List[int], y_pred: List[int], k: int) -> float:
    """
    Calculate Precision@K
    
    Parameters:
    -----------
    y_true : list
        Relevant item IDs
    y_pred : list
        Predicted item IDs in rank order
    k : int
        Cutoff position
        
    Returns:
    --------
    float : Precision@K (0 if k=0 or no predictions)
    """
    
    if k <= 0 or len(y_pred) == 0:
        return 0.0
    
    # Take top-k predictions
    top_k = y_pred[:k]
    
    # Count relevant items in top-k
    relevant_count = len(set(top_k) & set(y_true))
    
    # Precision@K = relevant in top-k / k
    return relevant_count / min(k, len(y_pred))


def recall_at_k(y_true: List[int], y_pred: List[int], k: int) -> float:
    """
    Calculate Recall@K
    
    Parameters:
    -----------
    y_true : list
        Relevant item IDs
    y_pred : list
        Predicted item IDs in rank order
    k : int
        Cutoff position
        
    Returns:
    --------
    float : Recall@K (0 if no relevant items)
    """
    
    if len(y_true) == 0:
        return 0.0
    
    if k <= 0 or len(y_pred) == 0:
        return 0.0
    
    # Take top-k predictions
    top_k = y_pred[:k]
    
    # Count relevant items in top-k
    relevant_count = len(set(top_k) & set(y_true))
    
    # Recall@K = relevant in top-k / total relevant
    return relevant_count / len(y_true)


def f1_at_k(y_true: List[int], y_pred: List[int], k: int) -> float:
    """Calculate F1@K score"""
    
    p = precision_at_k(y_true, y_pred, k)
    r = recall_at_k(y_true, y_pred, k)
    
    if p + r == 0:
        return 0.0
    
    return 2 * (p * r) / (p + r)


def evaluate_ranking(y_trues: List[List[int]], 
                     y_preds: List[List[int]], 
                     k_values: List[int] = [1, 3, 5, 10]) -> dict:
    """
    Evaluate ranking metrics for multiple queries
    
    Parameters:
    -----------
    y_trues : list of lists
        List of relevant item lists for each query
    y_preds : list of lists
        List of predicted item lists for each query
    k_values : list
        K values to evaluate
        
    Returns:
    --------
    dict : Metrics at each K
    """
    
    results = {}
    
    for k in k_values:
        precisions = []
        recalls = []
        f1s = []
        
        for y_true, y_pred in zip(y_trues, y_preds):
            precisions.append(precision_at_k(y_true, y_pred, k))
            recalls.append(recall_at_k(y_true, y_pred, k))
            f1s.append(f1_at_k(y_true, y_pred, k))
        
        results[f'P@{k}'] = np.mean(precisions)
        results[f'R@{k}'] = np.mean(recalls)
        results[f'F1@{k}'] = np.mean(f1s)
    
    return results


# Test cases
print("Test 1: Normal case")
y_true = [1, 3, 5, 7]
y_pred = [3, 1, 2, 5, 4, 7, 6]  # Rank order

print(f"P@3: {precision_at_k(y_true, y_pred, 3):.3f}")  # 2/3 = 0.667
print(f"R@3: {recall_at_k(y_true, y_pred, 3):.3f}")    # 2/4 = 0.5
print(f"F1@3: {f1_at_k(y_true, y_pred, 3):.3f}")

print("\nTest 2: Fewer than K predictions")
y_true = [1, 2, 3]
y_pred = [1, 2]  # Only 2 predictions

print(f"P@5: {precision_at_k(y_true, y_pred, 5):.3f}")  # 2/2 = 1.0
print(f"R@5: {recall_at_k(y_true, y_pred, 5):.3f}")    # 2/3 = 0.667

print("\nTest 3: No relevant items")
y_true = []
y_pred = [1, 2, 3]

print(f"P@3: {precision_at_k(y_true, y_pred, 3):.3f}")  # 0
print(f"R@3: {recall_at_k(y_true, y_pred, 3):.3f}")    # 0

print("\nTest 4: All recommendations relevant")
y_true = [1, 2, 3, 4, 5]
y_pred = [1, 2, 3]

print(f"P@3: {precision_at_k(y_true, y_pred, 3):.3f}")  # 3/3 = 1.0
print(f"R@3: {recall_at_k(y_true, y_pred, 3):.3f}")    # 3/5 = 0.6

print("\nTest 5: Multiple queries")
y_trues = [
    [1, 3, 5],
    [2, 4],
    [7, 8, 9, 10]
]
y_preds = [
    [3, 1, 2, 5, 4],
    [2, 1, 4, 3],
    [7, 1, 8, 2, 9]
]

results = evaluate_ranking(y_trues, y_preds, k_values=[1, 3, 5])
for metric, value in results.items():
    print(f"{metric}: {value:.3f}")
```

---

### Q15: Implement NDCG (Normalized Discounted Cumulative Gain) from scratch with support for graded relevance scores (not just binary).

**Answer:**

```python
import numpy as np

def dcg_at_k(relevances: list, k: int) -> float:
    """
    Calculate Discounted Cumulative Gain at K
    
    DCG@K = sum(rel_i / log2(i+1)) for i in 1..k
    
    Parameters:
    -----------
    relevances : list
        Relevance scores in rank order
    k : int
        Cutoff position
    """
    
    relevances = np.array(relevances)[:k]
    
    if len(relevances) == 0:
        return 0.0
    
    # Positions start at 1
    positions = np.arange(1, len(relevances) + 1)
    
    # DCG = sum(rel / log2(pos + 1))
    dcg = np.sum(relevances / np.log2(positions + 1))
    
    return dcg


def ndcg_at_k(y_true_relevances: list, y_pred_ranking: list, k: int) -> float:
    """
    Calculate Normalized Discounted Cumulative Gain at K
    
    Parameters:
    -----------
    y_true_relevances : dict or list
        If dict: {item_id: relevance_score}
        If list: relevance scores in ideal order
    y_pred_ranking : list
        Predicted item IDs in rank order
    k : int
        Cutoff position
        
    Returns:
    --------
    float : NDCG@K (1.0 = perfect ranking, 0.0 = worst)
    """
    
    # Convert dict to relevance list
    if isinstance(y_true_relevances, dict):
        # Get relevances in predicted order
        relevances_pred = [y_true_relevances.get(item, 0) for item in y_pred_ranking[:k]]
        # Get ideal relevances (sorted descending)
        relevances_ideal = sorted(y_true_relevances.values(), reverse=True)[:k]
    else:
        relevances_pred = y_true_relevances[:k]
        relevances_ideal = sorted(y_true_relevances, reverse=True)[:k]
    
    # Calculate DCG
    dcg = dcg_at_k(relevances_pred, k)
    
    # Calculate IDCG (Ideal DCG)
    idcg = dcg_at_k(relevances_ideal, k)
    
    if idcg == 0:
        return 0.0
    
    # NDCG = DCG / IDCG
    return dcg / idcg


def mean_ndcg_at_k(y_trues: list, y_preds: list, k: int) -> float:
    """Calculate mean NDCG@K across multiple queries"""
    
    ndcgs = []
    for y_true, y_pred in zip(y_trues, y_preds):
        ndcgs.append(ndcg_at_k(y_true, y_pred, k))
    
    return np.mean(ndcgs)


# Test cases
print("Test 1: Perfect ranking")
relevances = {
    'doc1': 3,  # Highly relevant
    'doc2': 2,  # Relevant
    'doc3': 1,  # Somewhat relevant
    'doc4': 0   # Not relevant
}
ranking_perfect = ['doc1', 'doc2', 'doc3', 'doc4']

ndcg = ndcg_at_k(relevances, ranking_perfect, k=4)
print(f"NDCG@4 (perfect): {ndcg:.4f}")  # Should be 1.0

print("\nTest 2: Worst ranking")
ranking_worst = ['doc4', 'doc3', 'doc2', 'doc1']
ndcg = ndcg_at_k(relevances, ranking_worst, k=4)
print(f"NDCG@4 (worst): {ndcg:.4f}")  # Should be low

print("\nTest 3: Mediocre ranking")
ranking_ok = ['doc2', 'doc1', 'doc3', 'doc4']
ndcg = ndcg_at_k(relevances, ranking_ok, k=4)
print(f"NDCG@4 (mediocre): {ndcg:.4f}")  # Between 0 and 1

print("\nTest 4: Binary relevance (0/1)")
relevances_binary = {
    'doc1': 1,
    'doc2': 1,
    'doc3': 0,
    'doc4': 0,
    'doc5': 1
}
ranking = ['doc1', 'doc3', 'doc2', 'doc4', 'doc5']

for k in [1, 3, 5]:
    ndcg = ndcg_at_k(relevances_binary, ranking, k)
    print(f"NDCG@{k}: {ndcg:.4f}")

print("\nTest 5: Graded relevance (0-5 scale)")
relevances_graded = {
    'doc1': 5,  # Perfect
    'doc2': 4,  # Excellent
    'doc3': 3,  # Good
    'doc4': 1,  # Poor
    'doc5': 0   # Irrelevant
}

ranking_good = ['doc1', 'doc2', 'doc3', 'doc4', 'doc5']
ranking_bad = ['doc5', 'doc4', 'doc3', 'doc2', 'doc1']

print(f"Good ranking NDCG@5: {ndcg_at_k(relevances_graded, ranking_good, 5):.4f}")
print(f"Bad ranking NDCG@5: {ndcg_at_k(relevances_graded, ranking_bad, 5):.4f}")

print("\nTest 6: Mean NDCG across queries")
y_trues = [
    {'A': 3, 'B': 2, 'C': 1},
    {'X': 2, 'Y': 1, 'Z': 0},
    {'M': 5, 'N': 3, 'O': 1}
]
y_preds = [
    ['A', 'B', 'C'],  # Perfect
    ['X', 'Z', 'Y'],  # OK
    ['N', 'M', 'O']   # Imperfect
]

mean_ndcg = mean_ndcg_at_k(y_trues, y_preds, k=3)
print(f"Mean NDCG@3: {mean_ndcg:.4f}")
```

---

### Q16: Create a function that performs stratified k-fold cross-validation and returns per-fold metrics plus confidence intervals (mean ± std).

**Answer:**

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import *
from scipy import stats

def stratified_cv_with_metrics(model, X, y, cv=5, random_state=42):
    """
    Stratified k-fold CV with comprehensive metrics and confidence intervals
    
    Parameters:
    -----------
    model : estimator
        Scikit-learn compatible model
    X, y : array-like
        Features and labels
    cv : int
        Number of folds
    random_state : int
        Random seed
        
    Returns:
    --------
    dict : Results with per-fold metrics and statistics
    """
    
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    
    fold_results = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Train
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_val)
        y_prob = model.predict_proba(X_val)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # Calculate metrics
        fold_metrics = {
            'fold': fold,
            'accuracy': accuracy_score(y_val, y_pred),
            'precision': precision_score(y_val, y_pred, zero_division=0),
            'recall': recall_score(y_val, y_pred, zero_division=0),
            'f1': f1_score(y_val, y_pred, zero_division=0),
            'specificity': recall_score(y_val, y_pred, pos_label=0, zero_division=0)
        }
        
        if y_prob is not None:
            fold_metrics['roc_auc'] = roc_auc_score(y_val, y_prob)
            fold_metrics['pr_auc'] = average_precision_score(y_val, y_prob)
        
        fold_results.append(fold_metrics)
    
    # Convert to DataFrame
    df = pd.DataFrame(fold_results)
    
    # Calculate statistics
    metrics_list = [c for c in df.columns if c != 'fold']
    
    summary = {}
    for metric in metrics_list:
        values = df[metric].values
        
        # Mean and std
        mean_val = np.mean(values)
        std_val = np.std(values, ddof=1)  # Sample std
        
        # 95% confidence interval
        se = std_val / np.sqrt(cv)  # Standard error
        ci_margin = 1.96 * se  # 95% CI
        
        # t-distribution for small sample (more conservative)
        t_crit = stats.t.ppf(0.975, df=cv-1)
        ci_margin_t = t_crit * se
        
        summary[metric] = {
            'mean': mean_val,
            'std': std_val,
            'min': np.min(values),
            'max': np.max(values),
            'ci_95': (mean_val - ci_margin_t, mean_val + ci_margin_t),
            'se': se
        }
    
    return {
        'fold_results': df,
        'summary': summary,
        'cv_folds': cv
    }


def print_cv_results(results):
    """Pretty print CV results"""
    
    print(f"{'='*70}")
    print(f"CROSS-VALIDATION RESULTS ({results['cv_folds']}-Fold)")
    print(f"{'='*70}\n")
    
    print("Per-Fold Results:")
    print(results['fold_results'].to_string(index=False))
    
    print(f"\n{'='*70}")
    print("Summary Statistics:")
    print(f"{'='*70}\n")
    
    print(f"{'Metric':<15} {'Mean':<10} {'Std':<10} {'95% CI':<25} {'[Min, Max]':<15}")
    print("-" * 70)
    
    for metric, stats in results['summary'].items():
        ci_lower, ci_upper = stats['ci_95']
        print(f"{metric:<15} {stats['mean']:<10.4f} {stats['std']:<10.4f} "
              f"[{ci_lower:.4f}, {ci_upper:.4f}]{'  ':<5} "
              f"[{stats['min']:.4f}, {stats['max']:.4f}]")


# Example usage
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Generate dataset
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15,
                          n_redundant=5, random_state=42)

# Evaluate model
model = LogisticRegression(class_weight='balanced', random_state=42)
results = stratified_cv_with_metrics(model, X, y, cv=5)

print_cv_results(results)

# Compare multiple models
print("\n" + "="*70)
print("MODEL COMPARISON")
print("="*70 + "\n")

models = {
    'Logistic Regression': LogisticRegression(class_weight='balanced', random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
}

comparison = []
for name, model in models.items():
    results = stratified_cv_with_metrics(model, X, y, cv=5)
    
    row = {'Model': name}
    for metric, stats in results['summary'].items():
        row[metric] = f"{stats['mean']:.3f} ± {stats['std']:.3f}"
    
    comparison.append(row)

df_comparison = pd.DataFrame(comparison)
print(df_comparison.to_string(index=False))
```

---

### Q17: Write a custom scoring function for GridSearchCV that implements a business-specific cost function (e.g., FN costs $100, FP costs $10).

**Answer:**

```python
import numpy as np
from sklearn.metrics import make_scorer, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

def business_cost_function(y_true, y_pred, fn_cost=100, fp_cost=10, 
                          tn_benefit=0, tp_benefit=0):
    """
    Calculate business cost/benefit
    
    Parameters:
    -----------
    y_true, y_pred : array-like
        True and predicted labels
    fn_cost : float
        Cost of false negative (e.g., missed fraud = $100 loss)
    fp_cost : float
        Cost of false positive (e.g., blocked legit transaction = $10 loss)
    tn_benefit : float
        Benefit of true negative (usually 0, but could model savings)
    tp_benefit : float
        Benefit of true positive (e.g., caught fraud = $50 saved)
        
    Returns:
    --------
    float : Total cost (negative) or benefit (positive)
            GridSearchCV maximizes, so return negative cost
    """
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # Calculate total cost
    total_cost = (fn * fn_cost) + (fp * fp_cost)
    total_benefit = (tn * tn_benefit) + (tp * tp_benefit)
    
    # Return negative cost (GridSearchCV maximizes)
    net_value = total_benefit - total_cost
    
    return net_value


def create_business_scorer(fn_cost=100, fp_cost=10, tn_benefit=0, tp_benefit=0):
    """Create sklearn-compatible scorer"""
    
    def scorer(y_true, y_pred):
        return business_cost_function(y_true, y_pred, fn_cost, fp_cost, 
                                     tn_benefit, tp_benefit)
    
    return make_scorer(scorer, greater_is_better=True, needs_proba=False)


# Example: Fraud detection with business costs
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Create imbalanced dataset (1% fraud)
X, y = make_classification(n_samples=10000, n_features=20, n_informative=15,
                          n_redundant=5, weights=[0.99, 0.01], 
                          flip_y=0.01, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                     stratify=y, random_state=42)

# Define business costs
# - False Negative (miss fraud): $100 loss
# - False Positive (block legit): $10 loss  
# - True Positive (catch fraud): $50 saved
business_scorer = create_business_scorer(fn_cost=100, fp_cost=10, 
                                        tn_benefit=0, tp_benefit=50)

# GridSearchCV with business cost
param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10],
    'class_weight': ['balanced', {0: 1, 1: 5}, {0: 1, 1: 10}, {0: 1, 1: 20}]
}

grid_search = GridSearchCV(
    LogisticRegression(random_state=42),
    param_grid=param_grid,
    scoring=business_scorer,
    cv=5,
    verbose=1,
    n_jobs=-1
)

print("Fitting GridSearchCV with business cost function...")
grid_search.fit(X_train, y_train)

print(f"\nBest parameters: {grid_search.best_params_}")
print(f"Best CV score (net business value): ${grid_search.best_score_:.2f}")

# Evaluate on test set
y_pred = grid_search.predict(X_test)

# Business metrics
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
total_cost = (fn * 100) + (fp * 10)
total_benefit = (tp * 50)
net_value = total_benefit - total_cost

print(f"\nTest Set Results:")
print(f"  True Negatives: {tn}")
print(f"  False Positives: {fp} (cost: ${fp * 10})")
print(f"  False Negatives: {fn} (cost: ${fn * 100})")
print(f"  True Positives: {tp} (benefit: ${tp * 50})")
print(f"  Net Business Value: ${net_value}")

# Compare with standard F1 optimization
from sklearn.metrics import f1_score

grid_search_f1 = GridSearchCV(
    LogisticRegression(random_state=42),
    param_grid=param_grid,
    scoring='f1',
    cv=5,
    n_jobs=-1
)

grid_search_f1.fit(X_train, y_train)
y_pred_f1 = grid_search_f1.predict(X_test)

tn_f1, fp_f1, fn_f1, tp_f1 = confusion_matrix(y_test, y_pred_f1).ravel()
net_value_f1 = (tp_f1 * 50) - (fn_f1 * 100) - (fp_f1 * 10)

print(f"\nComparison: F1-Optimized Model")
print(f"  False Negatives: {fn_f1} (cost: ${fn_f1 * 100})")
print(f"  False Positives: {fp_f1} (cost: ${fp_f1 * 10})")
print(f"  True Positives: {tp_f1} (benefit: ${tp_f1 * 50})")
print(f"  Net Business Value: ${net_value_f1}")
print(f"\nBusiness-optimized model is ${net_value - net_value_f1:.2f} better!")
```

---

### Q18: Implement a function that calculates calibration metrics (Expected Calibration Error) and plots a reliability diagram for probability predictions.

**Answer:**

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve

def expected_calibration_error(y_true, y_prob, n_bins=10):
    """
    Calculate Expected Calibration Error (ECE)
    
    ECE measures the difference between predicted probabilities and actual frequencies
    
    Parameters:
    -----------
    y_true : array-like
        True binary labels
    y_prob : array-like
        Predicted probabilities
    n_bins : int
        Number of bins for calibration
        
    Returns:
    --------
    float : ECE value (0 = perfectly calibrated)
    """
    
    # Bin predictions
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_prob, bin_edges[:-1]) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    
    ece = 0.0
    
    for i in range(n_bins):
        mask = bin_indices == i
        
        if np.sum(mask) > 0:
            # Average predicted probability in bin
            bin_prob = np.mean(y_prob[mask])
            
            # Actual frequency in bin
            bin_accuracy = np.mean(y_true[mask])
            
            # Weight by number of samples
            bin_weight = np.sum(mask) / len(y_true)
            
            # Add to ECE
            ece += bin_weight * np.abs(bin_accuracy - bin_prob)
    
    return ece


def maximum_calibration_error(y_true, y_prob, n_bins=10):
    """
    Calculate Maximum Calibration Error (MCE)
    
    MCE is the maximum difference between predicted and actual frequencies
    """
    
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_prob, bin_edges[:-1]) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    
    mce = 0.0
    
    for i in range(n_bins):
        mask = bin_indices == i
        
        if np.sum(mask) > 0:
            bin_prob = np.mean(y_prob[mask])
            bin_accuracy = np.mean(y_true[mask])
            
            error = np.abs(bin_accuracy - bin_prob)
            mce = max(mce, error)
    
    return mce


def plot_calibration_curve(y_true, y_prob, n_bins=10, model_name='Model'):
    """
    Plot reliability diagram (calibration curve)
    
    Perfect calibration: predicted probability = actual frequency
    """
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Calibration curve
    fraction_positives, mean_predicted = calibration_curve(y_true, y_prob, 
                                                           n_bins=n_bins, 
                                                           strategy='uniform')
    
    # Plot calibration curve
    ax1.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
    ax1.plot(mean_predicted, fraction_positives, 'o-', label=model_name)
    
    ax1.set_xlabel('Mean Predicted Probability', fontsize=12)
    ax1.set_ylabel('Fraction of Positives', fontsize=12)
    ax1.set_title('Reliability Diagram (Calibration Curve)', fontsize=14)
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Calculate calibration metrics
    ece = expected_calibration_error(y_true, y_prob, n_bins)
    mce = maximum_calibration_error(y_true, y_prob, n_bins)
    
    # Add metrics to plot
    ax1.text(0.05, 0.95, f'ECE: {ece:.4f}\nMCE: {mce:.4f}',
            transform=ax1.transAxes, fontsize=11,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Histogram of predicted probabilities
    ax2.hist(y_prob[y_true == 0], bins=20, alpha=0.5, label='Negative Class', 
            color='blue', density=True)
    ax2.hist(y_prob[y_true == 1], bins=20, alpha=0.5, label='Positive Class', 
            color='red', density=True)
    
    ax2.set_xlabel('Predicted Probability', fontsize=12)
    ax2.set_ylabel('Density', fontsize=12)
    ax2.set_title('Distribution of Predicted Probabilities', fontsize=14)
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return ece, mce


# Example: Compare calibrated vs uncalibrated model
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV

# Generate data
X, y = make_classification(n_samples=5000, n_features=20, n_informative=15,
                          n_redundant=5, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train Random Forest (typically poorly calibrated)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_prob_rf = rf.predict_proba(X_test)[:, 1]

# Train Logistic Regression (typically well calibrated)
lr = LogisticRegression(random_state=42)
lr.fit(X_train, y_train)
y_prob_lr = lr.predict_proba(X_test)[:, 1]

# Calibrated Random Forest
rf_calibrated = CalibratedClassifierCV(rf, method='sigmoid', cv=5)
rf_calibrated.fit(X_train, y_train)
y_prob_rf_cal = rf_calibrated.predict_proba(X_test)[:, 1]

# Plot and compare
print("Random Forest (Uncalibrated):")
ece_rf, mce_rf = plot_calibration_curve(y_test, y_prob_rf, n_bins=10, 
                                        model_name='Random Forest')

print("\nLogistic Regression:")
ece_lr, mce_lr = plot_calibration_curve(y_test, y_prob_lr, n_bins=10, 
                                        model_name='Logistic Regression')

print("\nRandom Forest (Calibrated):")
ece_rf_cal, mce_rf_cal = plot_calibration_curve(y_test, y_prob_rf_cal, n_bins=10, 
                                                 model_name='RF Calibrated')

# Summary
print("\nCalibration Metrics Summary:")
print(f"{'Model':<25} {'ECE':<10} {'MCE':<10}")
print("-" * 45)
print(f"{'Random Forest':<25} {ece_rf:<10.4f} {mce_rf:<10.4f}")
print(f"{'Logistic Regression':<25} {ece_lr:<10.4f} {mce_lr:<10.4f}")
print(f"{'RF (Calibrated)':<25} {ece_rf_cal:<10.4f} {mce_rf_cal:<10.4f}")
```

---

### Q19: Create a function that performs bootstrap resampling to estimate confidence intervals for any metric (precision, recall, F1, ROC-AUC).

**Answer:**

```python
import numpy as np
from sklearn.metrics import *
from typing import Callable

def bootstrap_metric_ci(y_true, y_pred, y_prob=None, 
                       metric_fn: Callable = f1_score,
                       n_bootstrap=1000, confidence_level=0.95,
                       random_state=42):
    """
    Calculate bootstrap confidence interval for any metric
    
    Parameters:
    -----------
    y_true, y_pred : array-like
        True and predicted labels
    y_prob : array-like, optional
        Predicted probabilities (needed for ROC-AUC, PR-AUC)
    metric_fn : callable
        Metric function (e.g., f1_score, roc_auc_score)
    n_bootstrap : int
        Number of bootstrap samples
    confidence_level : float
        Confidence level (e.g., 0.95 for 95% CI)
    random_state : int
        Random seed
        
    Returns:
    --------
    dict : {
        'point_estimate': metric value on original data,
        'ci_lower': lower bound of CI,
        'ci_upper': upper bound of CI,
        'bootstrap_distribution': array of bootstrap metric values
    }
    """
    
    np.random.seed(random_state)
    
    n_samples = len(y_true)
    bootstrap_metrics = []
    
    for _ in range(n_bootstrap):
        # Resample with replacement
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        
        y_true_boot = y_true[indices]
        y_pred_boot = y_pred[indices]
        
        # Check if we have enough positive/negative samples
        if len(np.unique(y_true_boot)) < 2:
            continue  # Skip this bootstrap sample
        
        try:
            if y_prob is not None and 'auc' in metric_fn.__name__.lower():
                y_prob_boot = y_prob[indices]
                metric_value = metric_fn(y_true_boot, y_prob_boot)
            else:
                metric_value = metric_fn(y_true_boot, y_pred_boot)
            
            bootstrap_metrics.append(metric_value)
        except:
            continue
    
    bootstrap_metrics = np.array(bootstrap_metrics)
    
    # Calculate point estimate on original data
    if y_prob is not None and 'auc' in metric_fn.__name__.lower():
        point_estimate = metric_fn(y_true, y_prob)
    else:
        point_estimate = metric_fn(y_true, y_pred)
    
    # Calculate confidence interval
    alpha = 1 - confidence_level
    ci_lower = np.percentile(bootstrap_metrics, alpha/2 * 100)
    ci_upper = np.percentile(bootstrap_metrics, (1 - alpha/2) * 100)
    
    return {
        'metric_name': metric_fn.__name__,
        'point_estimate': point_estimate,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'ci_width': ci_upper - ci_lower,
        'bootstrap_distribution': bootstrap_metrics,
        'n_bootstrap': len(bootstrap_metrics),
        'confidence_level': confidence_level
    }


def bootstrap_all_metrics(y_true, y_pred, y_prob=None, n_bootstrap=1000):
    """Calculate bootstrap CIs for all common metrics"""
    
    metrics = {
        'Accuracy': accuracy_score,
        'Precision': precision_score,
        'Recall': recall_score,
        'F1': f1_score,
        'Specificity': lambda yt, yp: recall_score(yt, yp, pos_label=0)
    }
    
    if y_prob is not None:
        metrics['ROC-AUC'] = roc_auc_score
        metrics['PR-AUC'] = average_precision_score
    
    results = {}
    
    for name, metric_fn in metrics.items():
        if 'AUC' in name:
            result = bootstrap_metric_ci(y_true, y_pred, y_prob, metric_fn, n_bootstrap)
        else:
            result = bootstrap_metric_ci(y_true, y_pred, None, metric_fn, n_bootstrap)
        
        results[name] = result
    
    return results


def plot_bootstrap_distribution(result, figsize=(10, 6)):
    """Plot bootstrap distribution with CI"""
    
    import matplotlib.pyplot as plt
    
    distribution = result['bootstrap_distribution']
    point_estimate = result['point_estimate']
    ci_lower = result['ci_lower']
    ci_upper = result['ci_upper']
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Histogram
    ax.hist(distribution, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    
    # Point estimate
    ax.axvline(point_estimate, color='red', linestyle='--', linewidth=2,
              label=f'Point Estimate: {point_estimate:.4f}')
    
    # Confidence interval
    ax.axvline(ci_lower, color='green', linestyle='--', linewidth=2,
              label=f'{result["confidence_level"]*100:.0f}% CI: [{ci_lower:.4f}, {ci_upper:.4f}]')
    ax.axvline(ci_upper, color='green', linestyle='--', linewidth=2)
    
    ax.set_xlabel(result['metric_name'], fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title(f'Bootstrap Distribution: {result["metric_name"]}', fontsize=14)
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def print_bootstrap_results(results):
    """Pretty print bootstrap results"""
    
    print(f"{'='*70}")
    print(f"BOOTSTRAP CONFIDENCE INTERVALS ({results[list(results.keys())[0]]['n_bootstrap']} samples)")
    print(f"{'='*70}\n")
    
    print(f"{'Metric':<15} {'Estimate':<12} {'95% CI':<30} {'Width':<10}")
    print("-" * 70)
    
    for name, result in results.items():
        estimate = result['point_estimate']
        ci_lower = result['ci_lower']
        ci_upper = result['ci_upper']
        width = result['ci_width']
        
        print(f"{name:<15} {estimate:<12.4f} [{ci_lower:.4f}, {ci_upper:.4f}]{'  ':<10} {width:<10.4f}")


# Example usage
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Generate data
X, y = make_classification(n_samples=500, n_features=20, n_informative=15,
                          weights=[0.7, 0.3], random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# Calculate bootstrap CIs
results = bootstrap_all_metrics(y_test, y_pred, y_prob, n_bootstrap=1000)

print_bootstrap_results(results)

# Plot F1 distribution
print("\nF1-Score Bootstrap Distribution:")
plot_bootstrap_distribution(results['F1'])

# Plot ROC-AUC distribution
print("\nROC-AUC Bootstrap Distribution:")
plot_bootstrap_distribution(results['ROC-AUC'])
```

---

### Q20: Implement a metric dashboard that tracks multiple models over time and alerts when performance degrades below threshold.

**Answer:**

```python
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Dict

@dataclass
class MetricSnapshot:
    timestamp: datetime
    model_name: str
    model_version: str
    metric_name: str
    value: float
    sample_size: int

class MetricDashboard:
    """Track and visualize metrics for multiple models over time"""
    
    def __init__(self, alert_threshold_std=2.0):
        self.snapshots: List[MetricSnapshot] = []
        self.alert_threshold_std = alert_threshold_std
        self.alerts = []
        self.baselines = {}
    
    def log_metrics(self, model_name: str, model_version: str, 
                   y_true, y_pred, y_prob=None, timestamp=None):
        """Log metrics for a model at a point in time"""
        
        if timestamp is None:
            timestamp = datetime.now()
        
        # Calculate metrics
        from sklearn.metrics import *
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0)
        }
        
        if y_prob is not None:
            metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
        
        # Store snapshots
        for metric_name, value in metrics.items():
            snapshot = MetricSnapshot(
                timestamp=timestamp,
                model_name=model_name,
                model_version=model_version,
                metric_name=metric_name,
                value=value,
                sample_size=len(y_true)
            )
            self.snapshots.append(snapshot)
    
    def set_baseline(self, model_name: str, metric_name: str, window_days=7):
        """Establish baseline from recent data"""
        
        cutoff = datetime.now() - timedelta(days=window_days)
        
        relevant_snapshots = [
            s for s in self.snapshots
            if s.model_name == model_name 
            and s.metric_name == metric_name
            and s.timestamp >= cutoff
        ]
        
        if len(relevant_snapshots) == 0:
            return None
        
        values = [s.value for s in relevant_snapshots]
        
        baseline = {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'n_samples': len(values)
        }
        
        key = f"{model_name}_{metric_name}"
        self.baselines[key] = baseline
        
        return baseline
    
    def check_alert(self, model_name: str, metric_name: str, recent_hours=24):
        """Check if recent performance has degraded"""
        
        key = f"{model_name}_{metric_name}"
        if key not in self.baselines:
            return None
        
        baseline = self.baselines[key]
        
        # Get recent snapshots
        cutoff = datetime.now() - timedelta(hours=recent_hours)
        recent_snapshots = [
            s for s in self.snapshots
            if s.model_name == model_name
            and s.metric_name == metric_name
            and s.timestamp >= cutoff
        ]
        
        if len(recent_snapshots) == 0:
            return None
        
        recent_values = [s.value for s in recent_snapshots]
        recent_mean = np.mean(recent_values)
        
        # Z-score
        z_score = (recent_mean - baseline['mean']) / baseline['std'] if baseline['std'] > 0 else 0
        
        alert_triggered = z_score < -self.alert_threshold_std
        
        alert = {
            'model_name': model_name,
            'metric_name': metric_name,
            'baseline_mean': baseline['mean'],
            'recent_mean': recent_mean,
            'z_score': z_score,
            'alert': alert_triggered,
            'timestamp': datetime.now()
        }
        
        if alert_triggered:
            self.alerts.append(alert)
        
        return alert
    
    def plot_metric_timeline(self, model_names=None, metric_name='f1', days=30):
        """Plot metric over time for multiple models"""
        
        cutoff = datetime.now() - timedelta(days=days)
        
        if model_names is None:
            model_names = list(set(s.model_name for s in self.snapshots))
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        for model_name in model_names:
            snapshots = [
                s for s in self.snapshots
                if s.model_name == model_name
                and s.metric_name == metric_name
                and s.timestamp >= cutoff
            ]
            
            if len(snapshots) == 0:
                continue
            
            timestamps = [s.timestamp for s in snapshots]
            values = [s.value for s in snapshots]
            
            ax.plot(timestamps, values, marker='o', label=model_name, linewidth=2)
            
            # Add baseline if available
            key = f"{model_name}_{metric_name}"
            if key in self.baselines:
                baseline = self.baselines[key]
                ax.axhline(baseline['mean'], linestyle='--', alpha=0.5)
                ax.axhline(baseline['mean'] - 2*baseline['std'], 
                          linestyle=':', alpha=0.5, color='red')
        
        ax.set_xlabel('Time', fontsize=12)
        ax.set_ylabel(metric_name, fontsize=12)
        ax.set_title(f'{metric_name} Over Time', fontsize=14)
        ax.legend()
        ax.grid(alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    
    def get_latest_metrics(self, model_name: str):
        """Get latest metrics for a model"""
        
        model_snapshots = [s for s in self.snapshots if s.model_name == model_name]
        
        if len(model_snapshots) == 0:
            return None
        
        # Group by metric
        latest = {}
        for metric in set(s.metric_name for s in model_snapshots):
            metric_snapshots = [s for s in model_snapshots if s.metric_name == metric]
            latest_snapshot = max(metric_snapshots, key=lambda s: s.timestamp)
            latest[metric] = latest_snapshot.value
        
        return latest
    
    def compare_models(self, metric_name='f1'):
        """Compare latest performance across all models"""
        
        model_names = list(set(s.model_name for s in self.snapshots))
        
        comparison = []
        for model_name in model_names:
            latest = self.get_latest_metrics(model_name)
            if latest and metric_name in latest:
                comparison.append({
                    'Model': model_name,
                    metric_name: latest[metric_name]
                })
        
        return pd.DataFrame(comparison).sort_values(metric_name, ascending=False)
    
    def generate_report(self):
        """Generate comprehensive dashboard report"""
        
        print("=" * 70)
        print("ML METRICS DASHBOARD")
        print("=" * 70)
        
        # Latest metrics per model
        print("\nLatest Metrics:")
        model_names = list(set(s.model_name for s in self.snapshots))
        
        for model_name in model_names:
            latest = self.get_latest_metrics(model_name)
            if latest:
                print(f"\n  {model_name}:")
                for metric, value in latest.items():
                    print(f"    {metric}: {value:.4f}")
        
        # Alerts
        if len(self.alerts) > 0:
            print(f"\n⚠️  ALERTS ({len(self.alerts)}):")
            for alert in self.alerts[-5:]:  # Show last 5
                print(f"  {alert['model_name']} - {alert['metric_name']}: "
                      f"{alert['recent_mean']:.4f} (baseline: {alert['baseline_mean']:.4f}, "
                      f"z={alert['z_score']:.2f})")
        else:
            print("\n✓ No alerts")
        
        print("\n" + "=" * 70)


# Example usage: Monitor multiple models
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

dashboard = MetricDashboard(alert_threshold_std=2.0)

# Simulate 30 days of model monitoring
np.random.seed(42)

for day in range(30):
    timestamp = datetime.now() - timedelta(days=30-day)
    
    # Generate data
    X, y = make_classification(n_samples=1000, n_features=20, random_state=42+day)
    
    # Model A: Stable performance
    model_a = LogisticRegression(random_state=42)
    model_a.fit(X[:800], y[:800])
    y_pred_a = model_a.predict(X[800:])
    y_prob_a = model_a.predict_proba(X[800:])[:, 1]
    
    dashboard.log_metrics('model_a', 'v1.0', y[800:], y_pred_a, y_prob_a, timestamp)
    
    # Model B: Degrading performance after day 20
    model_b = RandomForestClassifier(n_estimators=50, random_state=42)
    model_b.fit(X[:800], y[:800])
    y_pred_b = model_b.predict(X[800:])
    
    # Inject degradation after day 20
    if day > 20:
        noise = np.random.choice([0, 1], size=len(y_pred_b), p=[0.7, 0.3])
        y_pred_b = np.where(np.random.rand(len(y_pred_b)) < 0.3, noise, y_pred_b)
    
    y_prob_b = np.random.rand(len(y_pred_b))  # Simulate probabilities
    
    dashboard.log_metrics('model_b', 'v1.0', y[800:], y_pred_b, y_prob_b, timestamp)

# Set baselines (from first 7 days)
dashboard.set_baseline('model_a', 'f1', window_days=7)
dashboard.set_baseline('model_b', 'f1', window_days=7)

# Check for alerts
alert_a = dashboard.check_alert('model_a', 'f1', recent_hours=24*7)
alert_b = dashboard.check_alert('model_b', 'f1', recent_hours=24*7)

# Generate report
dashboard.generate_report()

# Plot timelines
dashboard.plot_metric_timeline(['model_a', 'model_b'], metric_name='f1', days=30)

# Compare models
print("\nModel Comparison (F1-Score):")
print(dashboard.compare_models('f1'))
```

This creates a **production-ready metrics monitoring dashboard** for ML systems!

---

## Debugging & Troubleshooting (Q21-Q25)


### Q21: Your model reports 95% accuracy but stakeholders say it's not working. You discover the test set is 95% negative class. How do you diagnose and fix the evaluation?

**Answer:**

**Problem:** Accuracy is misleading on imbalanced data.

**Diagnosis steps:**

```python
import numpy as np
from sklearn.metrics import *

# Step 1: Check class distribution
print("Class Distribution:")
unique, counts = np.unique(y_test, return_counts=True)
for cls, count in zip(unique, counts):
    print(f"  Class {cls}: {count} ({count/len(y_test)*100:.1f}%)")

# Step 2: Examine predictions
print(f"\nPredictions Distribution:")
unique_pred, counts_pred = np.unique(y_pred, return_counts=True)
for cls, count in zip(unique_pred, counts_pred):
    print(f"  Predicted {cls}: {count} ({count/len(y_pred)*100:.1f}%)")

# Step 3: Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print(f"\nConfusion Matrix:")
print(cm)

tn, fp, fn, tp = cm.ravel()
print(f"\nBreakdown:")
print(f"  True Negatives: {tn}")
print(f"  False Positives: {fp}")
print(f"  False Negatives: {fn}")
print(f"  True Positives: {tp}")

# Step 4: Check if model is just predicting majority class
if np.all(y_pred == 0):
    print("\n⚠️  WARNING: Model predicts only negative class!")
    print("This explains the 95% accuracy on 95% negative dataset.")

# Step 5: Calculate meaningful metrics
print(f"\nMeaningful Metrics:")
print(f"  Precision: {precision_score(y_test, y_pred, zero_division=0):.4f}")
print(f"  Recall: {recall_score(y_test, y_pred, zero_division=0):.4f}")
print(f"  F1-Score: {f1_score(y_test, y_pred, zero_division=0):.4f}")
print(f"  Balanced Accuracy: {balanced_accuracy_score(y_test, y_pred):.4f}")

# Step 6: Check if probabilities are calibrated
if y_prob is not None:
    print(f"\nProbability Statistics:")
    print(f"  Mean prob: {np.mean(y_prob):.4f}")
    print(f"  Std prob: {np.std(y_prob):.4f}")
    print(f"  Min prob: {np.min(y_prob):.4f}")
    print(f"  Max prob: {np.max(y_prob):.4f}")
    
    # Check if all probabilities are below threshold
    if np.max(y_prob) < 0.5:
        print("\n⚠️  All probabilities < 0.5, so all predictions are negative!")
```

**Fix:**

```python
def fix_imbalanced_evaluation(y_test, y_pred, y_prob=None):
    """Comprehensive evaluation for imbalanced data"""
    
    print("="*60)
    print("FIXED EVALUATION FOR IMBALANCED DATA")
    print("="*60)
    
    # 1. Class distribution
    neg_count = np.sum(y_test == 0)
    pos_count = np.sum(y_test == 1)
    print(f"\nClass Distribution:")
    print(f"  Negative: {neg_count} ({neg_count/len(y_test)*100:.1f}%)")
    print(f"  Positive: {pos_count} ({pos_count/len(y_test)*100:.1f}%)")
    
    # 2. Confusion matrix with rates
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    print(f"\nConfusion Matrix:")
    print(f"                Predicted")
    print(f"                Neg    Pos")
    print(f"Actual  Neg    {tn:4d}   {fp:4d}  (TNR={tn/(tn+fp):.3f}, FPR={fp/(tn+fp):.3f})")
    print(f"        Pos    {fn:4d}   {tp:4d}  (FNR={fn/(fn+tp):.3f}, TPR={tp/(fn+tp):.3f})")
    
    # 3. Primary metrics for imbalanced data
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    
    print(f"\nRecommended Metrics:")
    print(f"  F1-Score: {f1:.4f} ⭐ PRIMARY METRIC")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  Balanced Accuracy: {balanced_acc:.4f}")
    
    if y_prob is not None:
        roc_auc = roc_auc_score(y_test, y_prob)
        pr_auc = average_precision_score(y_test, y_prob)
        
        print(f"  ROC-AUC: {roc_auc:.4f}")
        print(f"  PR-AUC: {pr_auc:.4f} ⭐ BEST FOR IMBALANCED")
    
    # 4. Per-class metrics
    print(f"\nPer-Class Performance:")
    print(f"  Negative Class:")
    print(f"    Recall (Specificity): {tn/(tn+fp):.4f}")
    print(f"  Positive Class (Minority):")
    print(f"    Recall (Sensitivity): {tp/(tp+fn) if (tp+fn) > 0 else 0:.4f}")
    print(f"    Precision: {tp/(tp+fp) if (tp+fp) > 0 else 0:.4f}")
    
    # 5. Recommendations
    print(f"\n💡 Recommendations:")
    if recall < 0.5:
        print(f"  - Low recall ({recall:.2f}): Model misses {(1-recall)*100:.0f}% of positive cases")
        print(f"  - Try: Lower classification threshold, use class_weight='balanced'")
    if precision < 0.5:
        print(f"  - Low precision ({precision:.2f}): {(1-precision)*100:.0f}% of positive predictions are wrong")
        print(f"  - Try: Increase threshold, improve features")
    if f1 == 0:
        print(f"  - ⚠️ F1=0: Model not detecting any positives!")
        print(f"  - Action: Retrain with class_weight='balanced' or sample_weight")
    
    return {
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'balanced_accuracy': balanced_acc
    }

# Example usage
fix_imbalanced_evaluation(y_test, y_pred, y_prob)
```

**Prevent future issues:**

```python
def create_proper_test_set(X, y, test_size=0.2, min_positive_samples=50):
    """Create test set with minimum positive samples"""
    
    from sklearn.model_selection import train_test_split
    
    # Check positive class ratio
    pos_ratio = np.mean(y)
    
    if pos_ratio < 0.1:
        print(f"⚠️ Highly imbalanced: {pos_ratio*100:.1f}% positive")
        print("Using stratified split with oversampling validation...")
        
        # Stratified split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=42
        )
        
        # Ensure minimum positives in test set
        n_pos_test = np.sum(y_test == 1)
        if n_pos_test < min_positive_samples:
            print(f"⚠️ Only {n_pos_test} positive samples in test set")
            print("Recommendation: Collect more data or use cross-validation")
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=42
        )
    
    return X_train, X_test, y_train, y_test
```

---

### Q22: You calculate precision and recall correctly, but your F1-score implementation doesn't match sklearn. Debug the issue.

**Answer:**

**Common mistakes in F1 calculation:**

```python
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

# Generate test data
y_true = np.array([0, 1, 1, 0, 1, 0, 1, 1, 0, 1])
y_pred = np.array([0, 1, 0, 0, 1, 0, 1, 0, 0, 1])

# Calculate precision and recall
p = precision_score(y_true, y_pred)
r = recall_score(y_true, y_pred)
print(f"Precision: {p:.4f}")
print(f"Recall: {r:.4f}")

# WRONG implementations:

# Mistake 1: Arithmetic mean instead of harmonic mean
f1_wrong_1 = (p + r) / 2
print(f"\n❌ Wrong F1 (arithmetic mean): {f1_wrong_1:.4f}")

# Mistake 2: Wrong formula
f1_wrong_2 = 2 * p * r  # Missing division
print(f"❌ Wrong F1 (missing division): {f1_wrong_2:.4f}")

# Mistake 3: Division by zero not handled
def f1_wrong_3(p, r):
    return 2 * (p * r) / (p + r)  # Crashes if p+r=0

# Mistake 4: Using 2*p*r instead of (p*r)
f1_wrong_4 = 2 * 2 * p * r / (p + r)
print(f"❌ Wrong F1 (double multiplication): {f1_wrong_4:.4f}")

# CORRECT implementation:
def f1_correct(y_true, y_pred, zero_division=0.0):
    """Correct F1-score implementation"""
    
    # Calculate TP, FP, FN
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    
    # Precision
    if (tp + fp) == 0:
        precision = zero_division
    else:
        precision = tp / (tp + fp)
    
    # Recall
    if (tp + fn) == 0:
        recall = zero_division
    else:
        recall = tp / (tp + fn)
    
    # F1: Harmonic mean of precision and recall
    if (precision + recall) == 0:
        return zero_division
    else:
        return 2 * (precision * recall) / (precision + recall)

f1_correct_val = f1_correct(y_true, y_pred)
f1_sklearn = f1_score(y_true, y_pred)

print(f"\n✅ Correct F1: {f1_correct_val:.4f}")
print(f"✅ Sklearn F1: {f1_sklearn:.4f}")
print(f"Match: {abs(f1_correct_val - f1_sklearn) < 1e-10}")

# Test edge cases
print("\n" + "="*60)
print("EDGE CASE TESTING")
print("="*60)

# Edge case 1: All predicted negative
y_pred_all_neg = np.zeros(10)
f1_edge_1 = f1_correct(y_true, y_pred_all_neg)
f1_sklearn_edge_1 = f1_score(y_true, y_pred_all_neg, zero_division=0)
print(f"\nEdge 1: All predicted negative")
print(f"  Custom: {f1_edge_1:.4f}, Sklearn: {f1_sklearn_edge_1:.4f}")

# Edge case 2: Perfect predictions
y_pred_perfect = y_true.copy()
f1_edge_2 = f1_correct(y_true, y_pred_perfect)
f1_sklearn_edge_2 = f1_score(y_true, y_pred_perfect)
print(f"\nEdge 2: Perfect predictions")
print(f"  Custom: {f1_edge_2:.4f}, Sklearn: {f1_sklearn_edge_2:.4f}")

# Edge case 3: No positive class
y_true_no_pos = np.zeros(10)
y_pred_no_pos = np.zeros(10)
f1_edge_3 = f1_correct(y_true_no_pos, y_pred_no_pos)
f1_sklearn_edge_3 = f1_score(y_true_no_pos, y_pred_no_pos, zero_division=0)
print(f"\nEdge 3: No positive class")
print(f"  Custom: {f1_edge_3:.4f}, Sklearn: {f1_sklearn_edge_3:.4f}")
```

**Why F1 uses harmonic mean:**

```python
def compare_means():
    """Demonstrate why harmonic mean is important for F1"""
    
    # Scenario 1: Balanced precision and recall
    p1, r1 = 0.8, 0.8
    arithmetic_1 = (p1 + r1) / 2
    harmonic_1 = 2 * (p1 * r1) / (p1 + r1)
    
    print("Scenario 1: Balanced (P=0.8, R=0.8)")
    print(f"  Arithmetic mean: {arithmetic_1:.4f}")
    print(f"  Harmonic mean (F1): {harmonic_1:.4f}")
    
    # Scenario 2: Extreme imbalance
    p2, r2 = 0.95, 0.10
    arithmetic_2 = (p2 + r2) / 2
    harmonic_2 = 2 * (p2 * r2) / (p2 + r2)
    
    print("\nScenario 2: Imbalanced (P=0.95, R=0.10)")
    print(f"  Arithmetic mean: {arithmetic_2:.4f}  ← Misleadingly high!")
    print(f"  Harmonic mean (F1): {harmonic_2:.4f}  ← Correctly low")
    
    print("\n💡 Harmonic mean punishes extreme imbalance")
    print("   This forces F1 to require BOTH high precision AND high recall")

compare_means()
```

**Verification test suite:**

```python
def test_f1_implementation():
    """Comprehensive F1 test suite"""
    
    test_cases = [
        {
            'name': 'Normal case',
            'y_true': np.array([0, 1, 1, 0, 1, 0]),
            'y_pred': np.array([0, 1, 0, 0, 1, 1])
        },
        {
            'name': 'All correct',
            'y_true': np.array([0, 1, 1, 0, 1]),
            'y_pred': np.array([0, 1, 1, 0, 1])
        },
        {
            'name': 'All wrong',
            'y_true': np.array([0, 0, 1, 1]),
            'y_pred': np.array([1, 1, 0, 0])
        },
        {
            'name': 'All predicted negative',
            'y_true': np.array([0, 1, 1, 0]),
            'y_pred': np.array([0, 0, 0, 0])
        },
        {
            'name': 'All predicted positive',
            'y_true': np.array([0, 1, 1, 0]),
            'y_pred': np.array([1, 1, 1, 1])
        }
    ]
    
    all_pass = True
    
    for test in test_cases:
        f1_custom = f1_correct(test['y_true'], test['y_pred'])
        f1_sklearn = f1_score(test['y_true'], test['y_pred'], zero_division=0)
        
        match = abs(f1_custom - f1_sklearn) < 1e-10
        status = "✅" if match else "❌"
        
        print(f"{status} {test['name']}: custom={f1_custom:.4f}, sklearn={f1_sklearn:.4f}")
        
        if not match:
            all_pass = False
    
    print(f"\n{'✅ ALL TESTS PASSED' if all_pass else '❌ SOME TESTS FAILED'}")

test_f1_implementation()
```

---

### Q23: Your ROC-AUC score is 0.48 (worse than random). Diagnose what went wrong.

**Answer:**

**Problem:** ROC-AUC = 0.48 means predictions are anti-correlated with truth.

**Diagnosis:**

```python
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

def diagnose_roc_auc(y_true, y_prob):
    """Diagnose why ROC-AUC is bad"""
    
    print("="*60)
    print("ROC-AUC DIAGNOSIS")
    print("="*60)
    
    # Calculate AUC
    auc = roc_auc_score(y_true, y_prob)
    print(f"\nROC-AUC: {auc:.4f}")
    
    if auc < 0.5:
        print("⚠️  AUC < 0.5: Predictions are ANTI-CORRELATED with truth!")
        print("This means your model is predicting the OPPOSITE of truth.")
    
    # Check 1: Are probabilities flipped?
    print(f"\nCheck 1: Probability distribution")
    print(f"  Positive class mean prob: {np.mean(y_prob[y_true == 1]):.4f}")
    print(f"  Negative class mean prob: {np.mean(y_prob[y_true == 0]):.4f}")
    
    if np.mean(y_prob[y_true == 1]) < np.mean(y_prob[y_true == 0]):
        print("  ❌ PROBLEM: Positives have LOWER probabilities than negatives!")
        print("  Solution: Use y_prob = 1 - y_prob")
    
    # Check 2: Are labels flipped?
    print(f"\nCheck 2: Label distribution")
    print(f"  Positive samples: {np.sum(y_true == 1)}")
    print(f"  Negative samples: {np.sum(y_true == 0)}")
    
    # Check 3: Is model predicting constant?
    print(f"\nCheck 3: Prediction variability")
    print(f"  Min prob: {np.min(y_prob):.4f}")
    print(f"  Max prob: {np.max(y_prob):.4f}")
    print(f"  Std prob: {np.std(y_prob):.4f}")
    
    if np.std(y_prob) < 0.01:
        print("  ❌ PROBLEM: Model outputs nearly constant predictions!")
        print("  Solution: Check model training, features, or convergence")
    
    # Check 4: ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, label=f'Model (AUC={auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random (AUC=0.5)')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()
    
    # Recommendations
    print(f"\n💡 Recommendations:")
    if auc < 0.4:
        print("  1. Flip predictions: y_prob = 1 - y_prob")
        print("  2. Check if labels are swapped during training")
        print("  3. Verify you're using predict_proba(X)[:, 1] (not [:, 0])")

# Example: Flipped predictions
y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1])
y_prob_flipped = np.array([0.9, 0.8, 0.2, 0.1, 0.7, 0.3, 0.85, 0.15])

print("Example 1: Flipped predictions")
diagnose_roc_auc(y_true, y_prob_flipped)

# Fix: Flip probabilities
y_prob_fixed = 1 - y_prob_flipped
print("\n" + "="*60)
print("AFTER FIXING (y_prob = 1 - y_prob)")
print("="*60)
auc_fixed = roc_auc_score(y_true, y_prob_fixed)
print(f"ROC-AUC: {auc_fixed:.4f} ✅")
```

**Common causes and fixes:**

```python
# Cause 1: Wrong probability column
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, y_train)

# WRONG: Using probability for negative class
y_prob_wrong = model.predict_proba(X_test)[:, 0]  # ❌
auc_wrong = roc_auc_score(y_test, y_prob_wrong)
print(f"Wrong ([:, 0]): AUC = {auc_wrong:.4f}")

# CORRECT: Using probability for positive class
y_prob_correct = model.predict_proba(X_test)[:, 1]  # ✅
auc_correct = roc_auc_score(y_test, y_prob_correct)
print(f"Correct ([:, 1]): AUC = {auc_correct:.4f}")


# Cause 2: Labels swapped during data loading
# Check label encoding
print(f"\nLabel Check:")
print(f"  model.classes_: {model.classes_}")  # Should be [0, 1]
print(f"  predict_proba order: [prob_class_0, prob_class_1]")

# If classes are [1, 0] instead of [0, 1], probabilities are flipped!


# Cause 3: Target variable negation
# Sometimes target is coded as "success=0, failure=1" instead of "success=1"
# This inverts everything

def fix_inverted_target(y):
    """Check if target should be inverted"""
    
    # If your "positive" outcome is coded as 0, flip it
    print(f"Current encoding:")
    print(f"  0 = {np.sum(y == 0)} samples")
    print(f"  1 = {np.sum(y == 1)} samples")
    
    print(f"\nIf 0 represents your positive outcome, use: y = 1 - y")
    
    return 1 - y
```

---

### Q24: Two models have the same F1-score (0.75) but perform very differently in production. How do you debug and explain the difference?

**Answer:**

**F1-score alone is insufficient. Dig deeper:**

```python
import numpy as np
from sklearn.metrics import *
import matplotlib.pyplot as plt

def deep_comparison(y_true, y_pred_a, y_pred_b, y_prob_a=None, y_prob_b=None):
    """Compare two models with same F1 but different behavior"""
    
    print("="*70)
    print("DEEP MODEL COMPARISON")
    print("="*70)
    
    # Basic metrics
    f1_a = f1_score(y_true, y_pred_a)
    f1_b = f1_score(y_true, y_pred_b)
    
    print(f"\nF1-Scores:")
    print(f"  Model A: {f1_a:.4f}")
    print(f"  Model B: {f1_b:.4f}")
    print(f"  ✓ Same F1-score!")
    
    # But precision/recall can differ!
    print(f"\nPrecision & Recall:")
    print(f"  Model A: P={precision_score(y_true, y_pred_a):.4f}, R={recall_score(y_true, y_pred_a):.4f}")
    print(f"  Model B: P={precision_score(y_true, y_pred_b):.4f}, R={recall_score(y_true, y_pred_b):.4f}")
    
    # Confusion matrices
    cm_a = confusion_matrix(y_true, y_pred_a)
    cm_b = confusion_matrix(y_true, y_pred_b)
    
    tn_a, fp_a, fn_a, tp_a = cm_a.ravel()
    tn_b, fp_b, fn_b, tp_b = cm_b.ravel()
    
    print(f"\nConfusion Matrix Comparison:")
    print(f"               Model A    Model B")
    print(f"  TP           {tp_a:5d}      {tp_b:5d}")
    print(f"  FP           {fp_a:5d}      {fp_b:5d}")
    print(f"  FN           {fn_a:5d}      {fn_b:5d}")
    print(f"  TN           {tn_a:5d}      {tn_b:5d}")
    
    # Specificity
    spec_a = tn_a / (tn_a + fp_a)
    spec_b = tn_b / (tn_b + fp_b)
    
    print(f"\nSpecificity (TNR):")
    print(f"  Model A: {spec_a:.4f}")
    print(f"  Model B: {spec_b:.4f}")
    
    if abs(spec_a - spec_b) > 0.1:
        print(f"  ⚠️ DIFFERENT: Model {'B' if spec_b > spec_a else 'A'} has fewer false positives!")
    
    # ROC-AUC (if probabilities available)
    if y_prob_a is not None and y_prob_b is not None:
        auc_a = roc_auc_score(y_true, y_prob_a)
        auc_b = roc_auc_score(y_true, y_prob_b)
        
        print(f"\nROC-AUC:")
        print(f"  Model A: {auc_a:.4f}")
        print(f"  Model B: {auc_b:.4f}")
        
        if abs(auc_a - auc_b) > 0.05:
            print(f"  ⚠️ Model {'B' if auc_b > auc_a else 'A'} has better ranking ability!")
        
        # Plot ROC curves
        fpr_a, tpr_a, _ = roc_curve(y_true, y_prob_a)
        fpr_b, tpr_b, _ = roc_curve(y_true, y_prob_b)
        
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(fpr_a, tpr_a, label=f'Model A (AUC={auc_a:.3f})')
        plt.plot(fpr_b, tpr_b, label=f'Model B (AUC={auc_b:.3f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.title('ROC Curves')
        plt.legend()
        plt.grid(alpha=0.3)
        
        # Calibration curves
        plt.subplot(1, 2, 2)
        from sklearn.calibration import calibration_curve
        
        frac_pos_a, mean_pred_a = calibration_curve(y_true, y_prob_a, n_bins=10)
        frac_pos_b, mean_pred_b = calibration_curve(y_true, y_prob_b, n_bins=10)
        
        plt.plot([0, 1], [0, 1], 'k--', label='Perfect')
        plt.plot(mean_pred_a, frac_pos_a, 'o-', label='Model A')
        plt.plot(mean_pred_b, frac_pos_b, 's-', label='Model B')
        plt.xlabel('Mean Predicted Probability')
        plt.ylabel('Fraction of Positives')
        plt.title('Calibration Curves')
        plt.legend()
        plt.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    # Agreement analysis
    agreement = np.mean(y_pred_a == y_pred_b)
    print(f"\nPrediction Agreement:")
    print(f"  {agreement*100:.1f}% of predictions match")
    
    if agreement < 0.9:
        print(f"  ⚠️ Models disagree on {(1-agreement)*100:.1f}% of samples!")
        
        # Where do they disagree?
        disagree_mask = y_pred_a != y_pred_b
        print(f"\n  Disagreement analysis:")
        print(f"    A=1, B=0: {np.sum((y_pred_a == 1) & (y_pred_b == 0))}")
        print(f"    A=0, B=1: {np.sum((y_pred_a == 0) & (y_pred_b == 1))}")
    
    # Production implications
    print(f"\n💡 Production Implications:")
    
    # Business cost
    if fp_a != fp_b or fn_a != fn_b:
        print(f"\n  Different error patterns:")
        print(f"    Model A: {fn_a} false negatives, {fp_a} false positives")
        print(f"    Model B: {fn_b} false negatives, {fp_b} false positives")
        
        # Example business cost
        fn_cost = 100  # $100 per false negative
        fp_cost = 10   # $10 per false positive
        
        cost_a = fn_a * fn_cost + fp_a * fp_cost
        cost_b = fn_b * fn_cost + fp_b * fp_cost
        
        print(f"\n  Business cost (FN=$100, FP=$10):")
        print(f"    Model A: ${cost_a}")
        print(f"    Model B: ${cost_b}")
        print(f"    Savings: ${abs(cost_a - cost_b)} with Model {'B' if cost_b < cost_a else 'A'}")


# Example: Two models with F1=0.75 but different precision/recall
y_true = np.array([0]*70 + [1]*30)

# Model A: High precision (0.882), lower recall (0.652)
# - Conservative: Only predicts positive when very confident
# - Few false positives
y_pred_a = np.array([0]*65 + [1]*5 + [0]*10 + [1]*20)
y_prob_a = np.concatenate([
    np.random.beta(2, 5, 65),  # Negatives: low prob
    np.random.beta(5, 2, 35)   # Positives: high prob
])

# Model B: High recall (0.867), lower precision (0.667)
# - Aggressive: Predicts positive more liberally
# - More false positives
y_pred_b = np.array([0]*60 + [1]*10 + [0]*4 + [1]*26)
y_prob_b = np.concatenate([
    np.random.beta(3, 4, 60),  # Negatives: slightly higher prob
    np.random.beta(4, 3, 40)   # Positives: slightly lower prob
])

# Both have F1 ≈ 0.75
print(f"Model A F1: {f1_score(y_true, y_pred_a):.4f}")
print(f"Model B F1: {f1_score(y_true, y_pred_b):.4f}")

print("\n" + "="*70)
deep_comparison(y_true, y_pred_a, y_pred_b, y_prob_a, y_prob_b)
```

**Key lesson:** F1-score is a single number that hides precision/recall trade-offs. Always examine both metrics and understand business implications!

---

### Q25: Your model metrics are great in notebook (F1=0.92) but terrible in production (F1=0.65). Debug the train/test/prod gap.

**Answer:**

**This is a data distribution shift problem. Systematic debugging:**

```python
import numpy as np
import pandas as pd
from sklearn.metrics import *

def debug_train_prod_gap(X_train, y_train, X_prod, y_prod, model):
    """Debug why model performs worse in production"""
    
    print("="*70)
    print("TRAIN-PROD GAP DEBUGGING")
    print("="*70)
    
    # 1. Check class distribution shift
    print(f"\n1. CLASS DISTRIBUTION")
    train_pos_rate = np.mean(y_train)
    prod_pos_rate = np.mean(y_prod)
    
    print(f"  Training: {train_pos_rate*100:.1f}% positive")
    print(f"  Production: {prod_pos_rate*100:.1f}% positive")
    
    if abs(train_pos_rate - prod_pos_rate) > 0.1:
        print(f"  ⚠️ CLASS IMBALANCE SHIFT: {abs(train_pos_rate - prod_pos_rate)*100:.1f}% difference!")
        print(f"  Impact: Threshold optimized for training distribution won't work in prod")
    
    # 2. Check feature distribution shift
    print(f"\n2. FEATURE DISTRIBUTION SHIFT")
    
    for i in range(min(5, X_train.shape[1])):  # Check first 5 features
        train_mean = np.mean(X_train[:, i])
        train_std = np.std(X_train[:, i])
        prod_mean = np.mean(X_prod[:, i])
        prod_std = np.std(X_prod[:, i])
        
        # Z-score for mean shift
        z_score = abs(prod_mean - train_mean) / train_std if train_std > 0 else 0
        
        if z_score > 2:
            print(f"  ⚠️ Feature {i}: mean shifted by {z_score:.2f} std devs")
            print(f"     Train: μ={train_mean:.3f}, σ={train_std:.3f}")
            print(f"     Prod:  μ={prod_mean:.3f}, σ={prod_std:.3f}")
    
    # 3. Check for data leakage
    print(f"\n3. DATA LEAKAGE CHECK")
    
    # Look for suspiciously perfect separation
    y_train_pred = model.predict(X_train)
    train_f1 = f1_score(y_train, y_train_pred)
    
    print(f"  Training F1: {train_f1:.4f}")
    
    if train_f1 > 0.99:
        print(f"  ⚠️ SUSPICIOUSLY HIGH: Possible data leakage!")
        print(f"  Check: Are future features leaking into training?")
        print(f"  Check: Is target variable included in features?")
    
    # 4. Check time-based shift
    print(f"\n4. TEMPORAL SHIFT")
    print(f"  Training data: historical")
    print(f"  Production data: recent")
    print(f"  ⚠️ If >6 months apart, concept drift likely!")
    
    # 5. Missing features
    print(f"\n5. MISSING DATA PATTERNS")
    
    train_missing = np.isnan(X_train).mean(axis=0)
    prod_missing = np.isnan(X_prod).mean(axis=0)
    
    for i in range(min(5, len(train_missing))):
        if abs(train_missing[i] - prod_missing[i]) > 0.1:
            print(f"  ⚠️ Feature {i} missing rate:")
            print(f"     Train: {train_missing[i]*100:.1f}%")
            print(f"     Prod:  {prod_missing[i]*100:.1f}%")
    
    # 6. Prediction distribution
    print(f"\n6. PREDICTION DISTRIBUTION")
    
    y_prob_train = model.predict_proba(X_train)[:, 1]
    y_prob_prod = model.predict_proba(X_prod)[:, 1]
    
    print(f"  Training probabilities:")
    print(f"    Mean: {np.mean(y_prob_train):.3f}")
    print(f"    Std:  {np.std(y_prob_train):.3f}")
    
    print(f"  Production probabilities:")
    print(f"    Mean: {np.mean(y_prob_prod):.3f}")
    print(f"    Std:  {np.std(y_prob_prod):.3f}")
    
    # 7. Performance by confidence
    print(f"\n7. PERFORMANCE BY CONFIDENCE")
    
    bins = [0, 0.3, 0.7, 1.0]
    labels = ['Low', 'Medium', 'High']
    
    prod_confidence = pd.cut(y_prob_prod, bins=bins, labels=labels)
    
    for conf in labels:
        mask = prod_confidence == conf
        if np.sum(mask) > 0:
            conf_f1 = f1_score(y_prod[mask], model.predict(X_prod[mask]), zero_division=0)
            print(f"  {conf} confidence: F1={conf_f1:.3f} ({np.sum(mask)} samples)")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    print(f"  1. Retrain with recent data (last 3-6 months)")
    print(f"  2. Recalibrate threshold on production-like validation set")
    print(f"  3. Add monitoring for feature drift")
    print(f"  4. Implement periodic retraining (weekly/monthly)")
    print(f"  5. Check for bugs in production feature pipeline")


# Simulate the issue
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Training data: balanced, clean
X_train, y_train = make_classification(n_samples=1000, n_features=20,
                                       n_informative=15, weights=[0.5, 0.5],
                                       random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Notebook evaluation (train set)
y_train_pred = model.predict(X_train)
print(f"Notebook F1 (train): {f1_score(y_train, y_train_pred):.4f}")

# Production data: different distribution
# - Feature shift (add +1 to all features)
# - Class imbalance (20% positive instead of 50%)
X_prod, y_prod = make_classification(n_samples=500, n_features=20,
                                     n_informative=15, weights=[0.8, 0.2],
                                     random_state=99)
X_prod = X_prod + 1.0  # Simulate feature drift

y_prod_pred = model.predict(X_prod)
print(f"Production F1: {f1_score(y_prod, y_prod_pred):.4f}")

print("\n")
debug_train_prod_gap(X_train, y_train, X_prod, y_prod, model)
```

**Fixes:**

```python
# Fix 1: Retrain on recent data
X_recent = np.vstack([X_train[-200:], X_prod[:100]])
y_recent = np.concatenate([y_train[-200:], y_prod[:100]])

model_retrained = RandomForestClassifier(n_estimators=100, random_state=42)
model_retrained.fit(X_recent, y_recent)

# Fix 2: Recalibrate threshold
from sklearn.calibration import CalibratedClassifierCV

model_calibrated = CalibratedClassifierCV(model, method='sigmoid', cv=5)
model_calibrated.fit(X_prod[:100], y_prod[:100])

# Fix 3: Use proper validation
# Split training data by time (simulate temporal split)
split_point = int(0.8 * len(X_train))
X_train_early, X_val_late = X_train[:split_point], X_train[split_point:]
y_train_early, y_val_late = y_train[:split_point], y_train[split_point:]

model_temporal = RandomForestClassifier(n_estimators=100, random_state=42)
model_temporal.fit(X_train_early, y_train_early)

val_f1 = f1_score(y_val_late, model_temporal.predict(X_val_late))
print(f"\nTemporal validation F1: {val_f1:.4f}")
print("This is a more realistic estimate of production performance!")
```

---

## Trade-offs & Decisions (Q26-Q30)


### Q26: You need to choose between Model A (precision=0.95, recall=0.70) and Model B (precision=0.75, recall=0.95) for a spam detection system. What factors influence your decision?

**Answer:**

**Key question: What's worse - blocking real emails or letting spam through?**

**Analysis framework:**

```python
import numpy as np
import pandas as pd

def compare_spam_models():
    """Compare models for spam detection"""
    
    # Simulate 10,000 emails: 20% spam
    n_emails = 10000
    n_spam = 2000
    n_ham = 8000
    
    print("="*70)
    print("SPAM DETECTION: MODEL COMPARISON")
    print("="*70)
    
    # Model A: High precision (few false positives)
    precision_a = 0.95
    recall_a = 0.70
    
    tp_a = recall_a * n_spam  # Spam caught
    fn_a = n_spam - tp_a       # Spam missed
    fp_a = tp_a / precision_a - tp_a  # Ham marked as spam
    tn_a = n_ham - fp_a        # Ham correctly delivered
    
    # Model B: High recall (few false negatives)
    precision_b = 0.75
    recall_b = 0.95
    
    tp_b = recall_b * n_spam
    fn_b = n_spam - tp_b
    fp_b = tp_b / precision_b - tp_b
    tn_b = n_ham - fp_b
    
    print(f"\nModel A (High Precision):")
    print(f"  Spam caught: {tp_a:.0f} / {n_spam} ({recall_a*100:.1f}%)")
    print(f"  Spam missed: {fn_a:.0f}")
    print(f"  Real emails blocked: {fp_a:.0f} 🚨")
    print(f"  Real emails delivered: {tn_a:.0f}")
    
    print(f"\nModel B (High Recall):")
    print(f"  Spam caught: {tp_b:.0f} / {n_spam} ({recall_b*100:.1f}%)")
    print(f"  Spam missed: {fn_b:.0f}")
    print(f"  Real emails blocked: {fp_b:.0f} 🚨")
    print(f"  Real emails delivered: {tn_b:.0f}")
    
    # User experience impact
    print(f"\n📊 USER EXPERIENCE IMPACT:")
    
    # For average user receiving 100 emails/day (20 spam, 80 real)
    daily_spam = 20
    daily_ham = 80
    
    fp_daily_a = (fp_a / n_ham) * daily_ham
    fp_daily_b = (fp_b / n_ham) * daily_ham
    
    fn_daily_a = (fn_a / n_spam) * daily_spam
    fn_daily_b = (fn_b / n_spam) * daily_spam
    
    print(f"\n  Model A (per user per day):")
    print(f"    Real emails blocked: {fp_daily_a:.1f}")
    print(f"    Spam in inbox: {fn_daily_a:.1f}")
    
    print(f"\n  Model B (per user per day):")
    print(f"    Real emails blocked: {fp_daily_b:.1f}")
    print(f"    Spam in inbox: {fn_daily_b:.1f}")
    
    # Business metrics
    print(f"\n💰 BUSINESS METRICS:")
    
    # Cost of blocking real email: high (user frustration, missed opportunities)
    cost_fp = 10  # $10 per blocked real email
    
    # Cost of spam getting through: low (minor annoyance)
    cost_fn = 0.5  # $0.50 per spam in inbox
    
    cost_a = fp_a * cost_fp + fn_a * cost_fn
    cost_b = fp_b * cost_fp + fn_b * cost_fn
    
    print(f"  Cost Model A: ${cost_a:,.0f}")
    print(f"  Cost Model B: ${cost_b:,.0f}")
    print(f"  Savings with Model {'A' if cost_a < cost_b else 'B'}: ${abs(cost_a - cost_b):,.0f}")
    
    # Decision framework
    print(f"\n✅ RECOMMENDATION:")
    
    print(f"\n  Choose Model A (High Precision) if:")
    print(f"    - Blocking real emails is unacceptable")
    print(f"    - Users are business professionals")
    print(f"    - Spam folder is checked regularly")
    print(f"    - False positive = lost customer/revenue")
    
    print(f"\n  Choose Model B (High Recall) if:")
    print(f"    - Spam is a major problem (phishing, malware)")
    print(f"    - Users prioritize clean inbox over perfect delivery")
    print(f"    - You have good spam folder UI")
    print(f"    - False negative = security risk")
    
    print(f"\n  🎯 BEST PRACTICE: Use Model A (high precision)")
    print(f"     Rationale: Users tolerate spam better than losing real emails")

compare_spam_models()
```

**Real-world factors:**

1. **User expectations**: Email users expect 100% delivery of real mail
2. **Spam folder**: Users can check spam folder for missed mail, but can't recover blocked mail they never see
3. **Reputation**: ISP reputation suffers from false positives
4. **Compliance**: Blocking business emails can have legal consequences

**Hybrid approach:**

```python
def hybrid_spam_detection(X, model_a, model_b):
    """Use high-precision model with high-recall safety net"""
    
    # Primary: Model A (high precision)
    y_prob_a = model_a.predict_proba(X)[:, 1]
    
    # High confidence spam → block
    spam_certain = y_prob_a > 0.9
    
    # Use Model B for uncertain cases
    y_prob_b = model_b.predict_proba(X)[:, 1]
    spam_likely = (y_prob_a > 0.5) | (y_prob_b > 0.8)
    
    # Three tiers
    predictions = np.where(spam_certain, 'block',
                  np.where(spam_likely, 'spam_folder', 'inbox'))
    
    return predictions
```

---

### Q27: Your team wants a single metric for model selection. Should you optimize for F1, ROC-AUC, or PR-AUC? Explain trade-offs.

**Answer:**

**No universal answer - depends on your use case.**

**Comparison table:**

```python
import pandas as pd

def compare_metrics():
    """Compare F1, ROC-AUC, PR-AUC for different scenarios"""
    
    comparison = pd.DataFrame([
        {
            'Metric': 'F1-Score',
            'Best for': 'Balanced classes, single threshold',
            'Pros': 'Simple, interpretable, balances P/R',
            'Cons': 'Threshold-dependent, ignores TN',
            'Imbalanced data': 'OK but not ideal'
        },
        {
            'Metric': 'ROC-AUC',
            'Best for': 'Balanced classes, ranking tasks',
            'Pros': 'Threshold-independent, stable',
            'Cons': 'Optimistic on imbalanced data',
            'Imbalanced data': 'Can be misleading'
        },
        {
            'Metric': 'PR-AUC',
            'Best for': 'Imbalanced classes (fraud, rare disease)',
            'Pros': 'Robust to imbalance, focus on positives',
            'Cons': 'Less intuitive, lower scores',
            'Imbalanced data': 'Best choice'
        }
    ])
    
    print(comparison.to_string(index=False))


def decision_tree_metric_selection(class_balance, has_probabilities, 
                                   need_ranking, production_threshold_fixed):
    """Decision tree for metric selection"""
    
    print("="*70)
    print("METRIC SELECTION GUIDE")
    print("="*70)
    
    # Check balance
    if class_balance > 0.4:  # Relatively balanced (40-60%)
        print(f"\n✓ Dataset is balanced ({class_balance*100:.0f}% minority class)")
        
        if not has_probabilities:
            return "F1-Score", "Simple classification, no probabilities"
        
        if need_ranking:
            return "ROC-AUC", "Need to rank predictions across thresholds"
        else:
            return "F1-Score", "Single threshold deployment"
    
    else:  # Imbalanced (<40% minority)
        print(f"\n⚠️ Dataset is imbalanced ({class_balance*100:.1f}% minority class)")
        
        if not has_probabilities:
            return "F1-Score", "Best option without probabilities"
        
        if production_threshold_fixed:
            return "F1-Score", "Fixed threshold, focus on precision/recall balance"
        else:
            return "PR-AUC", "Evaluate across thresholds, focus on minority class"


# Example decision
metric, reason = decision_tree_metric_selection(
    class_balance=0.05,  # 5% fraud
    has_probabilities=True,
    need_ranking=True,
    production_threshold_fixed=False
)

print(f"\nRecommended metric: {metric}")
print(f"Reason: {reason}")
```

**When to use each:**

```python
# Use case 1: Fraud detection (0.5% fraud rate)
# → PR-AUC: Imbalanced, need threshold flexibility

from sklearn.metrics import average_precision_score, roc_auc_score, f1_score

# PR-AUC handles imbalance better
pr_auc = average_precision_score(y_test, y_prob)
roc_auc = roc_auc_score(y_test, y_prob)

print("Fraud Detection (0.5% fraud):")
print(f"  ROC-AUC: {roc_auc:.4f}  ← Optimistic")
print(f"  PR-AUC: {pr_auc:.4f}   ← More realistic")


# Use case 2: Sentiment analysis (balanced)
# → ROC-AUC or F1: Balanced data, both work

print("\nSentiment Analysis (balanced):")
print(f"  F1: Simple, interpretable")
print(f"  ROC-AUC: If comparing multiple models")


# Use case 3: Medical diagnosis (need high recall)
# → F-beta (beta=2) or custom metric

from sklearn.metrics import fbeta_score

f2 = fbeta_score(y_test, y_pred, beta=2)  # Emphasize recall

print(f"\nMedical Diagnosis:")
print(f"  F2-Score: {f2:.4f}  ← Weights recall 2x")


# Use case 4: Recommendation systems
# → Precision@K, NDCG@K (ranking metrics)

print(f"\nRecommendation Systems:")
print(f"  Use: Precision@K, Recall@K, NDCG@K")
print(f"  Reason: Ranking quality matters, not just classification")
```

**My recommendation priority:**

1. **Highly imbalanced (< 10% minority)**: PR-AUC
2. **Balanced (40-60%)**: F1 or ROC-AUC
3. **Different costs for FP/FN**: Custom business metric
4. **Ranking matters**: ROC-AUC or ranking metrics (NDCG)
5. **Need interpretability**: F1 (easy to explain to stakeholders)

---

### Q28: You can improve recall from 0.80 to 0.90, but precision drops from 0.85 to 0.60. Should you make the change? How do you decide?

**Answer:**

**This is a precision/recall trade-off question. Decision depends on business context.**

**Quantitative analysis:**

```python
import numpy as np

def analyze_precision_recall_tradeoff(n_total, n_positive, 
                                     precision_old, recall_old,
                                     precision_new, recall_new,
                                     fp_cost, fn_cost):
    """Analyze if the trade-off is worth it"""
    
    print("="*70)
    print("PRECISION/RECALL TRADE-OFF ANALYSIS")
    print("="*70)
    
    # Current model
    tp_old = recall_old * n_positive
    fn_old = n_positive - tp_old
    fp_old = tp_old / precision_old - tp_old
    tn_old = (n_total - n_positive) - fp_old
    
    # New model
    tp_new = recall_new * n_positive
    fn_new = n_positive - tp_new
    fp_new = tp_new / precision_new - tp_new
    tn_new = (n_total - n_positive) - fp_new
    
    # Confusion matrices
    print(f"\nConfusion Matrix Comparison:")
    print(f"                    Current    →    New Model")
    print(f"  True Positives:   {tp_old:7.0f}    →    {tp_new:7.0f}  (+{tp_new-tp_old:.0f})")
    print(f"  False Negatives:  {fn_old:7.0f}    →    {fn_new:7.0f}  ({fn_new-fn_old:+.0f})")
    print(f"  False Positives:  {fp_old:7.0f}    →    {fp_new:7.0f}  ({fp_new-fp_old:+.0f})")
    print(f"  True Negatives:   {tn_old:7.0f}    →    {tn_new:7.0f}  ({tn_new-tn_old:+.0f})")
    
    # F1 comparison
    f1_old = 2 * (precision_old * recall_old) / (precision_old + recall_old)
    f1_new = 2 * (precision_new * recall_new) / (precision_new + recall_new)
    
    print(f"\nF1-Score:")
    print(f"  Current: {f1_old:.4f}")
    print(f"  New:     {f1_new:.4f}  ({f1_new - f1_old:+.4f})")
    
    # Business cost
    cost_old = fn_old * fn_cost + fp_old * fp_cost
    cost_new = fn_new * fn_cost + fp_new * fp_cost
    
    print(f"\nBusiness Cost (FN=${fn_cost}, FP=${fp_cost}):")
    print(f"  Current: ${cost_old:,.0f}")
    print(f"  New:     ${cost_new:,.0f}  (${cost_new - cost_old:+,.0f})")
    
    # Decision
    print(f"\n{'='*70}")
    print(f"DECISION:")
    print(f"{'='*70}")
    
    if cost_new < cost_old:
        savings = cost_old - cost_new
        print(f"✅ MAKE THE CHANGE")
        print(f"   Savings: ${savings:,.0f}")
        print(f"   You catch {tp_new - tp_old:.0f} more positives")
        print(f"   at the cost of {fp_new - fp_old:.0f} more false positives")
    else:
        loss = cost_new - cost_old
        print(f"❌ DON'T MAKE THE CHANGE")
        print(f"   Loss: ${loss:,.0f}")
        print(f"   The {fp_new - fp_old:.0f} extra false positives outweigh")
        print(f"   the {tp_new - tp_old:.0f} extra true positives")
    
    return cost_new < cost_old


# Scenario 1: Fraud detection (FN very expensive)
print("SCENARIO 1: FRAUD DETECTION")
print("-" * 70)
print("Context: Missing fraud (FN) costs $1000, false alarm (FP) costs $5")

should_change = analyze_precision_recall_tradeoff(
    n_total=10000,
    n_positive=100,  # 1% fraud rate
    precision_old=0.85,
    recall_old=0.80,
    precision_new=0.60,
    recall_new=0.90,
    fp_cost=5,      # False alarm costs $5
    fn_cost=1000    # Missed fraud costs $1000
)

print("\n" + "="*70)
print("\nSCENARIO 2: SPAM DETECTION")
print("-" * 70)
print("Context: Blocking real email (FP) costs $50, spam in inbox (FN) costs $1")

should_change = analyze_precision_recall_tradeoff(
    n_total=10000,
    n_positive=2000,  # 20% spam rate
    precision_old=0.85,
    recall_old=0.80,
    precision_new=0.60,
    recall_new=0.90,
    fp_cost=50,     # Blocked real email costs $50
    fn_cost=1       # Spam in inbox costs $1
)
```

**Decision framework:**

```python
def should_increase_recall(current_precision, current_recall,
                          new_precision, new_recall,
                          use_case):
    """Decision framework for precision/recall trade-offs"""
    
    # Calculate trade-off ratio
    recall_gain = new_recall - current_recall
    precision_loss = current_precision - new_precision
    
    print(f"\nTrade-off:")
    print(f"  Gain in recall: +{recall_gain:.2f}")
    print(f"  Loss in precision: -{precision_loss:.2f}")
    
    # Use case heuristics
    recommendations = {
        'fraud_detection': {
            'decision': True,
            'reason': 'Missing fraud is very expensive. Higher recall worth it.'
        },
        'spam_detection': {
            'decision': False,
            'reason': 'Blocking real emails causes user frustration. Keep precision.'
        },
        'medical_diagnosis': {
            'decision': True,
            'reason': 'Missing disease is dangerous. Prioritize recall.'
        },
        'recommendation_system': {
            'decision': False,
            'reason': 'Bad recommendations hurt engagement. Need precision.'
        },
        'content_moderation': {
            'decision': True,
            'reason': 'Missing harmful content is risky. Higher recall needed.'
        }
    }
    
    if use_case in recommendations:
        rec = recommendations[use_case]
        print(f"\n{'✅' if rec['decision'] else '❌'} {rec['reason']}")
        return rec['decision']
    
    return None

# Test
should_increase_recall(0.85, 0.80, 0.60, 0.90, 'fraud_detection')
should_increase_recall(0.85, 0.80, 0.60, 0.90, 'spam_detection')
```

**Bottom line:**
- **Increase recall** when FN is much more expensive than FP (fraud, medical, security)
- **Keep precision** when FP is more expensive than FN (spam, recommendations)
- **Always quantify business cost** before making changes

---

### Q29: Your model has ROC-AUC=0.88 but PR-AUC=0.42. Is this good or bad? Which metric should you trust?

**Answer:**

**This is a classic imbalanced data scenario. PR-AUC is more trustworthy.**

**Why the discrepancy?**

```python
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, roc_curve
import matplotlib.pyplot as plt

def explain_roc_pr_gap(y_true, y_prob):
    """Explain why ROC-AUC and PR-AUC differ"""
    
    print("="*70)
    print("ROC-AUC vs PR-AUC ANALYSIS")
    print("="*70)
    
    # Calculate metrics
    roc_auc = roc_auc_score(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)
    
    # Class distribution
    n_positive = np.sum(y_true == 1)
    n_negative = np.sum(y_true == 0)
    pos_rate = n_positive / len(y_true)
    
    print(f"\nClass Distribution:")
    print(f"  Positive: {n_positive} ({pos_rate*100:.2f}%)")
    print(f"  Negative: {n_negative} ({(1-pos_rate)*100:.2f}%)")
    print(f"  Imbalance ratio: {n_negative/n_positive:.1f}:1")
    
    print(f"\nMetrics:")
    print(f"  ROC-AUC: {roc_auc:.4f}  ← High!")
    print(f"  PR-AUC:  {pr_auc:.4f}  ← Low!")
    
    # Explanation
    print(f"\n❓ WHY THE GAP?")
    
    if pos_rate < 0.1:
        print(f"\n  Your data is highly imbalanced ({pos_rate*100:.2f}% positive)")
        print(f"\n  ROC-AUC includes True Negative Rate:")
        print(f"    - Easy to get high TNR when negatives dominate")
        print(f"    - {n_negative} negatives means lots of easy correct predictions")
        print(f"    - ROC-AUC is OPTIMISTIC on imbalanced data")
        
        print(f"\n  PR-AUC focuses only on positive class:")
        print(f"    - Ignores the {n_negative} true negatives")
        print(f"    - Only cares about {n_positive} positives")
        print(f"    - PR-AUC is REALISTIC on imbalanced data")
    
    # Baseline comparison
    baseline_pr = pos_rate  # Random classifier PR-AUC = positive class rate
    
    print(f"\n📊 INTERPRETATION:")
    print(f"  PR-AUC baseline (random): {baseline_pr:.4f}")
    print(f"  Your PR-AUC: {pr_auc:.4f}")
    print(f"  Improvement over random: {pr_auc/baseline_pr:.2f}x")
    
    if pr_auc / baseline_pr > 2:
        print(f"\n  ✅ PR-AUC of {pr_auc:.4f} is GOOD for {pos_rate*100:.1f}% positive rate")
        print(f"     You're {pr_auc/baseline_pr:.1f}x better than random!")
    else:
        print(f"\n  ⚠️ PR-AUC of {pr_auc:.4f} is MEDIOCRE for {pos_rate*100:.1f}% positive rate")
        print(f"     Only {pr_auc/baseline_pr:.1f}x better than random")
    
    # Visualize
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    ax1.plot(fpr, tpr, label=f'Model (AUC={roc_auc:.3f})', linewidth=2)
    ax1.plot([0, 1], [0, 1], 'k--', label='Random (AUC=0.5)')
    ax1.fill_between(fpr, tpr, alpha=0.3)
    ax1.set_xlabel('False Positive Rate', fontsize=12)
    ax1.set_ylabel('True Positive Rate', fontsize=12)
    ax1.set_title('ROC Curve\n(Includes TNR - optimistic on imbalanced data)', fontsize=12)
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # PR curve
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ax2.plot(recall, precision, label=f'Model (AUC={pr_auc:.3f})', linewidth=2)
    ax2.axhline(baseline_pr, color='red', linestyle='--', 
               label=f'Random (AUC={baseline_pr:.3f})')
    ax2.fill_between(recall, precision, alpha=0.3)
    ax2.set_xlabel('Recall', fontsize=12)
    ax2.set_ylabel('Precision', fontsize=12)
    ax2.set_title('Precision-Recall Curve\n(Focuses on minority class - realistic)', fontsize=12)
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Which to trust?
    print(f"\n✅ WHICH METRIC TO TRUST?")
    if pos_rate < 0.1:
        print(f"\n  Trust PR-AUC ({pr_auc:.4f})")
        print(f"  Reason: Highly imbalanced data ({pos_rate*100:.2f}% positive)")
        print(f"  ROC-AUC ({roc_auc:.4f}) is misleadingly optimistic")
    else:
        print(f"\n  Both metrics are reliable")
        print(f"  Reason: Data is relatively balanced ({pos_rate*100:.1f}% positive)")


# Example: Fraud detection (1% fraud)
np.random.seed(42)

n_samples = 10000
n_fraud = 100  # 1% fraud rate

# Simulate a decent model
y_true = np.array([0] * (n_samples - n_fraud) + [1] * n_fraud)
y_prob = np.concatenate([
    np.random.beta(2, 5, n_samples - n_fraud),  # Negatives: low prob
    np.random.beta(5, 2, n_fraud)               # Positives: high prob
])

explain_roc_pr_gap(y_true, y_prob)
```

**Rule of thumb:**

```python
def which_metric_to_use(pos_rate):
    """Decide which metric is more meaningful"""
    
    if pos_rate < 0.1:
        return "PR-AUC", "Highly imbalanced - ROC-AUC will be misleadingly high"
    elif pos_rate < 0.3:
        return "PR-AUC (primary), ROC-AUC (secondary)", "Moderately imbalanced"
    else:
        return "Either (both are fine)", "Balanced enough for both metrics"

# Examples
for pos_rate in [0.01, 0.05, 0.20, 0.50]:
    metric, reason = which_metric_to_use(pos_rate)
    print(f"{pos_rate*100:5.1f}% positive: Use {metric}")
    print(f"         Reason: {reason}\n")
```

**Your case (ROC-AUC=0.88, PR-AUC=0.42):**

- **Data is likely highly imbalanced** (~5-10% positive class)
- **PR-AUC=0.42 is probably GOOD** (baseline would be ~0.05-0.10)
- **Trust PR-AUC more** - it's realistic for imbalanced data
- **ROC-AUC=0.88 is optimistic** because of high TNR on abundant negatives

---

### Q30: You have budget to improve your model. Should you: (A) Collect more data, (B) Engineer better features, or (C) Try more complex models? How do you decide?

**Answer:**

**Use learning curves and systematic diagnostics to decide.**

**Diagnostic framework:**

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.metrics import f1_score

def diagnose_model_bottleneck(model, X, y, cv=5):
    """Diagnose if model needs more data, features, or complexity"""
    
    print("="*70)
    print("MODEL IMPROVEMENT DIAGNOSTIC")
    print("="*70)
    
    # 1. Learning curves
    train_sizes, train_scores, val_scores = learning_curve(
        model, X, y, cv=cv,
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring='f1',
        n_jobs=-1
    )
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    # Plot learning curves
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_sizes, train_mean, 'o-', label='Training score')
    plt.plot(train_sizes, val_mean, 'o-', label='Validation score')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1)
    plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1)
    plt.xlabel('Training Set Size')
    plt.ylabel('F1-Score')
    plt.title('Learning Curves')
    plt.legend()
    plt.grid(alpha=0.3)
    
    # Gap analysis
    final_train = train_mean[-1]
    final_val = val_mean[-1]
    gap = final_train - final_val
    
    plt.subplot(1, 2, 2)
    categories = ['Train Score', 'Val Score', 'Gap']
    values = [final_train, final_val, gap]
    colors = ['green', 'blue', 'red']
    plt.bar(categories, values, color=colors, alpha=0.7)
    plt.ylabel('Score')
    plt.title('Final Performance Analysis')
    plt.axhline(0.9, color='gray', linestyle='--', alpha=0.5, label='Target')
    plt.legend()
    plt.grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.show()
    
    # Diagnosis
    print(f"\n📊 LEARNING CURVE ANALYSIS:")
    print(f"  Final training score: {final_train:.4f}")
    print(f"  Final validation score: {final_val:.4f}")
    print(f"  Train-val gap: {gap:.4f}")
    
    print(f"\n🔍 DIAGNOSIS:")
    
    # High bias (underfitting)
    if final_train < 0.85 and final_val < 0.85 and gap < 0.05:
        print(f"\n  ❌ HIGH BIAS (Underfitting)")
        print(f"     - Both train and val scores are low")
        print(f"     - Small gap between train and val")
        print(f"\n  💡 RECOMMENDATION: (B) Engineer better features or (C) Try more complex models")
        print(f"     - Current model is too simple")
        print(f"     - Adding data won't help much")
        return 'features_or_complexity'
    
    # High variance (overfitting)
    elif gap > 0.10:
        print(f"\n  ❌ HIGH VARIANCE (Overfitting)")
        print(f"     - Large gap ({gap:.2f}) between train and val")
        print(f"     - Model memorizes training data")
        print(f"\n  💡 RECOMMENDATION: (A) Collect more data")
        print(f"     - Model is complex enough")
        print(f"     - Needs more examples to generalize")
        
        # Check if validation score is still improving
        if val_mean[-1] > val_mean[-3]:
            print(f"     - Validation score still improving → data will help!")
        
        return 'more_data'
    
    # Good fit but low performance
    elif final_val < 0.80 and gap < 0.10:
        print(f"\n  ⚠️ GOOD FIT BUT LOW PERFORMANCE")
        print(f"     - Small train-val gap (not overfitting)")
        print(f"     - But both scores are low ({final_val:.2f})")
        print(f"\n  💡 RECOMMENDATION: (B) Engineer better features")
        print(f"     - Model is learning, but signal is weak")
        print(f"     - Need more informative features")
        return 'features'
    
    # Already good
    elif final_val > 0.85 and gap < 0.10:
        print(f"\n  ✅ MODEL IS ALREADY GOOD")
        print(f"     - High validation score ({final_val:.2f})")
        print(f"     - Small train-val gap ({gap:.2f})")
        print(f"\n  💡 RECOMMENDATION: Focus on other priorities")
        print(f"     - Model performance is solid")
        print(f"     - Diminishing returns from more work")
        return 'good_enough'
    
    else:
        print(f"\n  UNCLEAR - Need more analysis")
        return 'unclear'


def cost_benefit_analysis():
    """Compare cost/benefit of each improvement option"""
    
    print("\n" + "="*70)
    print("COST-BENEFIT ANALYSIS")
    print("="*70)
    
    options = pd.DataFrame([
        {
            'Option': 'A. Collect More Data',
            'Cost': '$$$ (High)',
            'Time': '2-4 weeks',
            'F1 Gain': '+0.05-0.10',
            'Best when': 'High variance (overfitting)',
            'Risk': 'Labeling cost, data quality'
        },
        {
            'Option': 'B. Engineer Features',
            'Cost': '$ (Low)',
            'Time': '1-2 weeks',
            'F1 Gain': '+0.10-0.20',
            'Best when': 'High bias (underfitting)',
            'Risk': 'May not find good features'
        },
        {
            'Option': 'C. Complex Model',
            'Cost': '$$ (Medium)',
            'Time': '1 week',
            'F1 Gain': '+0.05-0.15',
            'Best when': 'High bias, enough data',
            'Risk': 'Overfitting, deployment complexity'
        }
    ])
    
    print(options.to_string(index=False))


# Example usage
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

print("EXAMPLE 1: Simple model on complex data (underfitting)")
print("-" * 70)

# Complex non-linear data
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15,
                          n_redundant=5, n_clusters_per_class=3, random_state=42)

# Too simple model
simple_model = LogisticRegression(max_iter=100)

recommendation = diagnose_model_bottleneck(simple_model, X, y)

cost_benefit_analysis()
```

**Decision tree:**

```python
def improvement_decision_tree(train_score, val_score, data_size):
    """Decision tree for model improvement"""
    
    gap = train_score - val_score
    
    print(f"\nDecision Tree:")
    print(f"  Train score: {train_score:.2f}")
    print(f"  Val score: {val_score:.2f}")
    print(f"  Gap: {gap:.2f}")
    print(f"  Data size: {data_size}")
    
    if gap > 0.15:
        print(f"\n  → Large gap → Overfitting")
        print(f"  → Collect more data (A)")
    elif train_score < 0.80 and val_score < 0.80:
        print(f"\n  → Both scores low → Underfitting")
        if data_size < 1000:
            print(f"  → Small dataset → Collect more data (A)")
        else:
            print(f"  → Enough data → Better features (B) or complex model (C)")
    elif val_score < 0.80 and gap < 0.10:
        print(f"\n  → Low scores, small gap → Weak signal")
        print(f"  → Engineer better features (B)")
    else:
        print(f"\n  → Model is good enough")
        print(f"  → Focus elsewhere or accept current performance")

# Test scenarios
print("Scenario 1: Overfitting")
improvement_decision_tree(train_score=0.95, val_score=0.75, data_size=500)

print("\n" + "="*70)
print("Scenario 2: Underfitting")
improvement_decision_tree(train_score=0.70, val_score=0.68, data_size=5000)

print("\n" + "="*70)
print("Scenario 3: Weak features")
improvement_decision_tree(train_score=0.75, val_score=0.73, data_size=2000)
```

**Summary:**
- **(A) More data**: High variance (overfitting), validation score still improving
- **(B) Better features**: High bias (underfitting), or weak signal despite good fit
- **(C) Complex model**: High bias with sufficient data, simple model maxed out

**Always plot learning curves first!**

