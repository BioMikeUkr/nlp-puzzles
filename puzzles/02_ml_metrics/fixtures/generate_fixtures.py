"""
Generate fixture data for ML Metrics module.
Run this script to create all input and expected output files.
"""
import numpy as np
import pandas as pd
import json
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)

# Set random seed for reproducibility
np.random.seed(42)

def generate_binary_classification_data():
    """Generate realistic binary classification data (imbalanced: 80% class 0, 20% class 1)"""
    n_samples = 1000
    n_class_0 = 800
    n_class_1 = 200

    # Generate features (5 columns) with some class separation
    features_0 = np.random.randn(n_class_0, 5) * 0.8
    features_1 = np.random.randn(n_class_1, 5) * 0.8 + 1.5  # Shift for separation

    features = np.vstack([features_0, features_1])
    true_labels = np.array([0] * n_class_0 + [1] * n_class_1)

    # Shuffle the data
    shuffle_idx = np.random.permutation(n_samples)
    features = features[shuffle_idx]
    true_labels = true_labels[shuffle_idx]

    # Generate realistic predictions
    # Model is good but not perfect (90% accuracy for class 0, 75% for class 1)
    predicted_labels = []
    predicted_probabilities = []

    for i, true_label in enumerate(true_labels):
        if true_label == 0:
            # 90% correct prediction for class 0
            if np.random.rand() < 0.90:
                pred_label = 0
                pred_prob = np.random.beta(8, 2)  # Probability skewed toward 0
            else:
                pred_label = 1
                pred_prob = np.random.beta(2, 8)  # Probability skewed toward 1
        else:
            # 75% correct prediction for class 1
            if np.random.rand() < 0.75:
                pred_label = 1
                pred_prob = np.random.beta(2, 8)  # Probability skewed toward 1
            else:
                pred_label = 0
                pred_prob = np.random.beta(8, 2)  # Probability skewed toward 0

        predicted_labels.append(pred_label)
        # Convert to probability of class 1
        predicted_probabilities.append(1 - pred_prob if pred_label == 0 else pred_prob)

    # Create DataFrame
    df = pd.DataFrame(
        features,
        columns=['feature_1', 'feature_2', 'feature_3', 'feature_4', 'feature_5']
    )
    df['true_label'] = true_labels
    df['predicted_label'] = predicted_labels
    df['predicted_probability'] = predicted_probabilities

    return df


def generate_multiclass_data():
    """Generate realistic multiclass classification data (3 classes)"""
    n_samples = 500
    n_per_class = n_samples // 3

    # Generate features with class-specific patterns
    features_0 = np.random.randn(n_per_class, 5) * 0.8 + np.array([0, 0, 0, 0, 0])
    features_1 = np.random.randn(n_per_class, 5) * 0.8 + np.array([2, 0, 1, 0, 1])
    features_2 = np.random.randn(n_per_class + (n_samples - 3*n_per_class), 5) * 0.8 + np.array([0, 2, 0, 2, 0])

    features = np.vstack([features_0, features_1, features_2])
    true_labels = np.array([0] * n_per_class + [1] * n_per_class +
                          [2] * (n_per_class + (n_samples - 3*n_per_class)))

    # Shuffle
    shuffle_idx = np.random.permutation(n_samples)
    features = features[shuffle_idx]
    true_labels = true_labels[shuffle_idx]

    # Generate predictions (80% accuracy overall)
    predicted_labels = []
    prob_class_0 = []
    prob_class_1 = []
    prob_class_2 = []

    for i, true_label in enumerate(true_labels):
        if np.random.rand() < 0.80:
            # Correct prediction
            pred_label = true_label
        else:
            # Wrong prediction (random other class)
            other_classes = [c for c in [0, 1, 2] if c != true_label]
            pred_label = np.random.choice(other_classes)

        # Generate probabilities with correct class having highest probability
        probs = np.random.dirichlet([1, 1, 1])  # Random probabilities
        # Boost the predicted class probability
        probs[pred_label] += 0.5
        probs = probs / probs.sum()  # Renormalize

        predicted_labels.append(pred_label)
        prob_class_0.append(probs[0])
        prob_class_1.append(probs[1])
        prob_class_2.append(probs[2])

    # Create DataFrame
    df = pd.DataFrame(
        features,
        columns=['feature_1', 'feature_2', 'feature_3', 'feature_4', 'feature_5']
    )
    df['true_label'] = true_labels
    df['predicted_label'] = predicted_labels
    df['prob_class_0'] = prob_class_0
    df['prob_class_1'] = prob_class_1
    df['prob_class_2'] = prob_class_2

    return df


def calculate_expected_metrics(df):
    """Calculate expected metrics for validation"""
    y_true = df['true_label'].values
    y_pred = df['predicted_label'].values
    y_prob = df['predicted_probability'].values

    metrics = {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred)),
        'recall': float(recall_score(y_true, y_pred)),
        'f1': float(f1_score(y_true, y_pred)),
        'roc_auc': float(roc_auc_score(y_true, y_prob)),
        'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
        'tp': int(((y_true == 1) & (y_pred == 1)).sum()),
        'fp': int(((y_true == 0) & (y_pred == 1)).sum()),
        'fn': int(((y_true == 1) & (y_pred == 0)).sum()),
        'tn': int(((y_true == 0) & (y_pred == 0)).sum())
    }

    return metrics


def main():
    """Generate all fixture files"""
    import os

    # Get the fixtures directory
    fixtures_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(fixtures_dir, 'input')
    expected_dir = os.path.join(fixtures_dir, 'expected')

    # Create directories if they don't exist
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(expected_dir, exist_ok=True)

    print("Generating binary classification data...")
    binary_df = generate_binary_classification_data()
    binary_path = os.path.join(input_dir, 'classification_data.csv')
    binary_df.to_csv(binary_path, index=False)
    print(f"✓ Created {binary_path}")
    print(f"  Shape: {binary_df.shape}")
    print(f"  Class distribution: {binary_df['true_label'].value_counts().to_dict()}")

    print("\nGenerating multiclass data...")
    multiclass_df = generate_multiclass_data()
    multiclass_path = os.path.join(input_dir, 'multiclass_data.csv')
    multiclass_df.to_csv(multiclass_path, index=False)
    print(f"✓ Created {multiclass_path}")
    print(f"  Shape: {multiclass_df.shape}")
    print(f"  Class distribution: {multiclass_df['true_label'].value_counts().to_dict()}")

    print("\nCalculating expected metrics...")
    metrics = calculate_expected_metrics(binary_df)
    metrics_path = os.path.join(expected_dir, 'metrics_results.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"✓ Created {metrics_path}")
    print(f"  Metrics: accuracy={metrics['accuracy']:.4f}, "
          f"precision={metrics['precision']:.4f}, "
          f"recall={metrics['recall']:.4f}, "
          f"f1={metrics['f1']:.4f}")

    print("\n✅ All fixtures generated successfully!")
    print("\nTo use these fixtures, load them in your notebooks:")
    print("  df = pd.read_csv('../../fixtures/input/classification_data.csv')")


if __name__ == '__main__':
    main()
