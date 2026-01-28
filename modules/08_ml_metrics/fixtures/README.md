# Fixtures for ML Metrics Module

This directory contains test data for the ML Metrics module.

## Generating Fixtures

To generate all fixture files, run:

```bash
cd /home/biomike/BioMike/nlp-puzzles/modules/08_ml_metrics/fixtures
python generate_fixtures.py
```

This will create:

### Input Files (`input/`)

1. **classification_data.csv** (1000 rows)
   - Binary classification dataset (imbalanced: 80% class 0, 20% class 1)
   - Columns: `feature_1` through `feature_5`, `true_label`, `predicted_label`, `predicted_probability`
   - Realistic predictions: ~90% accuracy on class 0, ~75% on class 1
   - Use for: confusion matrix, precision, recall, F1, ROC-AUC

2. **multiclass_data.csv** (500 rows)
   - Multi-class classification (3 classes: 0, 1, 2)
   - Columns: `feature_1` through `feature_5`, `true_label`, `predicted_label`, `prob_class_0`, `prob_class_1`, `prob_class_2`
   - ~80% overall accuracy
   - Use for: macro/micro/weighted averaging, multi-class confusion matrix

### Expected Files (`expected/`)

1. **metrics_results.json**
   - Pre-calculated metrics for `classification_data.csv`
   - Includes: accuracy, precision, recall, F1, ROC-AUC
   - Also includes: confusion matrix elements (TP, FP, TN, FN)
   - Use for: validating student calculations in tasks

## File Structure

```
fixtures/
├── README.md                          # This file
├── generate_fixtures.py               # Script to generate all data
├── input/                             # Input datasets
│   ├── classification_data.csv        # Binary classification data
│   └── multiclass_data.csv            # Multi-class data
├── expected/                          # Expected outputs for validation
│   └── metrics_results.json           # Expected metrics
└── edge_cases/                        # Edge case datasets (future)
    └── (to be added)
```

## Usage in Notebooks

```python
import pandas as pd

# Load binary classification data
df = pd.read_csv('../../fixtures/input/classification_data.csv')

# Load multiclass data
df_multi = pd.read_csv('../../fixtures/input/multiclass_data.csv')

# Load expected metrics for validation
import json
with open('../../fixtures/expected/metrics_results.json', 'r') as f:
    expected = json.load(f)
```

## Data Characteristics

### Binary Classification Data
- **Total samples**: 1000
- **Class distribution**:
  - Class 0: 800 samples (80%)
  - Class 1: 200 samples (20%)
- **Model performance**:
  - Overall accuracy: ~87%
  - Class 0 accuracy: ~90%
  - Class 1 accuracy (recall): ~75%
- **Imbalance challenge**: Demonstrates why accuracy alone is insufficient

### Multiclass Data
- **Total samples**: 500
- **Classes**: 3 (approximately balanced)
- **Overall accuracy**: ~80%
- **Use cases**:
  - Macro/micro/weighted averaging
  - Per-class metrics
  - Multi-class confusion matrices
