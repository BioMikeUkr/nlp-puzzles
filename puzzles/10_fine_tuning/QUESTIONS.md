# Deep Dive Q&A - Fine-tuning Sentence Transformers

> 30 questions covering fine-tuning strategies, implementation patterns, and production best practices

## Architecture & Design (Q1-Q10)

### Q1: When should you fine-tune instead of using a pre-trained model?

**Answer:**

**Decision Matrix:**

| Factor | Use Pre-trained | Fine-tune |
|--------|----------------|-----------|
| **Domain** | General (news, web) | Specialized (medical, legal) |
| **Data availability** | <500 pairs | >5K pairs |
| **Performance** | Accuracy >0.8 | Accuracy <0.7 |
| **Time/budget** | Limited | Flexible |
| **Vocabulary overlap** | High with training data | Low/specialized terms |

**Example scenarios:**

**Use pre-trained (all-MiniLM-L6-v2):**
```python
# General customer support tickets
ticket = "How do I reset my password?"
# Generic model understands "password", "reset" well
# Accuracy: 85% (good enough)
```

**Fine-tune:**
```python
# Medical diagnoses
query = "Patient presents with acute myocardial infarction"
# Generic model: treats as generic text
# Fine-tuned model: understands medical relationships
# Improvement: 65% → 89% accuracy

# Legal documents
query = "breach of fiduciary duty in corporate governance"
# Generic: 70% accuracy
# Fine-tuned on legal corpus: 91% accuracy
```

**ROI calculation:**
```python
# Cost of fine-tuning
gpu_hours = 2  # A100
cost_per_hour = 2
total_cost = 4

# Improvement in retrieval accuracy
queries_per_day = 10000
accuracy_improvement = 0.15  # 70% → 85%
fewer_failures = 10000 * 0.15  # 1500 queries/day

# If each failure costs $1 in support time
savings_per_day = 1500
roi = 1500 / 4  # 375x in one day
```

---

### Q2: What loss function should you use for different tasks?

**Answer:**

**Loss Function Selection:**

**1. MultipleNegativesRankingLoss (Recommended for RAG/Retrieval):**

```python
from sentence_transformers import losses

train_loss = losses.MultipleNegativesRankingLoss(model)

# Use when:
# ✓ Retrieval/search task
# ✓ (query, relevant_doc) pairs
# ✓ Want other batch items as negatives

# Data format:
train_examples = [
    InputExample(texts=['query1', 'relevant_doc1']),
    InputExample(texts=['query2', 'relevant_doc2']),
    # Other items in batch serve as negatives
]

# Why it works:
# - Efficient: uses batch for negatives (no need to sample)
# - Scales well: larger batch = more negatives
# - Best for asymmetric tasks (query ≠ document)
```

**2. ContrastiveLoss (Binary classification):**

```python
train_loss = losses.ContrastiveLoss(model)

# Use when:
# ✓ Have explicit positive/negative pairs
# ✓ Binary similarity labels (0 or 1)
# ✓ Symmetric task (sentence1 ≈ sentence2)

# Data format:
train_examples = [
    InputExample(texts=['sent1', 'sent2'], label=1.0),  # Similar
    InputExample(texts=['sent1', 'sent3'], label=0.0),  # Dissimilar
]
```

**3. TripletLoss (Ranking):**

```python
train_loss = losses.TripletLoss(model)

# Use when:
# ✓ Have anchor-positive-negative triplets
# ✓ Want to maximize margin between similar/dissimilar
# ✓ Ranking task

# Data format:
train_examples = [
    InputExample(texts=['anchor', 'positive', 'negative'])
]
```

**Comparison on retrieval task:**

```python
# Dataset: 10K query-document pairs
# Evaluation metric: nDCG@10

results = {
    'MultipleNegativesRankingLoss': {
        'nDCG@10': 0.82,
        'training_time': '45 min',
        'data_required': '10K pairs'
    },
    'ContrastiveLoss': {
        'nDCG@10': 0.76,
        'training_time': '60 min',
        'data_required': '20K pairs (10K pos + 10K neg)'
    },
    'TripletLoss': {
        'nDCG@10': 0.79,
        'training_time': '90 min',
        'data_required': '30K triplets'
    }
}
```

---

### Q3: How much training data do you need for fine-tuning?

**Answer:**

**Data Requirements by Task:**

| Task Complexity | Minimum | Recommended | Optimal |
|----------------|---------|-------------|---------|
| **Simple (2-3 categories)** | 500 pairs | 2K pairs | 5K pairs |
| **Medium (5-10 categories)** | 2K pairs | 5K pairs | 20K pairs |
| **Complex (>20 categories)** | 5K pairs | 20K pairs | 100K+ pairs |

**Empirical results:**

```python
# Experiment: Fine-tune on support ticket classification
# Base model: all-MiniLM-L6-v2
# Task: Classify into 8 categories

training_sizes = [100, 500, 1000, 2000, 5000, 10000, 50000]
results = {
    100:    {'accuracy': 0.62, 'improvement': 0.02},  # Minimal gain
    500:    {'accuracy': 0.71, 'improvement': 0.11},  # Noticeable
    1000:   {'accuracy': 0.78, 'improvement': 0.18},  # Good
    2000:   {'accuracy': 0.83, 'improvement': 0.23},  # Great
    5000:   {'accuracy': 0.87, 'improvement': 0.27},  # Excellent
    10000:  {'accuracy': 0.89, 'improvement': 0.29},  # Diminishing returns
    50000:  {'accuracy': 0.90, 'improvement': 0.30},  # Marginal gain
}

# Sweet spot: 2K-5K pairs for most tasks
```

**Data augmentation strategies:**

```python
# 1. Paraphrasing with back-translation
def augment_with_paraphrasing(text):
    # Use back-translation or T5 paraphrasing
    paraphrase = paraphrase_model(text)
    return paraphrase

# 2. Easy data mining (BM25 + filtering)
def mine_training_pairs(corpus, queries):
    """Find pseudo-positive pairs"""
    pairs = []

    for query in queries:
        # BM25 retrieval
        candidates = bm25_search(query, corpus, top_k=10)

        # Use cross-encoder to filter
        scores = cross_encoder.predict([(query, c) for c in candidates])

        # High score = good training pair
        for cand, score in zip(candidates, scores):
            if score > 0.8:  # High confidence
                pairs.append((query, cand))

    return pairs
```

---

### Q4: How do you evaluate fine-tuned models?

**Answer:**

**1. Information Retrieval Metrics (for RAG/Search):**

```python
from sentence_transformers.evaluation import InformationRetrievalEvaluator

# Setup
queries = {
    'q1': 'How to reset password?',
    'q2': 'What is pricing?',
}

corpus = {
    'd1': 'Password reset guide...',
    'd2': 'Pricing: $99/month...',
}

relevant_docs = {
    'q1': {'d1', 'd3'},  # Multiple relevant docs for q1
    'q2': {'d2'}
}

# Evaluate
evaluator = InformationRetrievalEvaluator(queries, corpus, relevant_docs)
results = evaluator(model, output_path='results/')

# Metrics returned:
{
    'ndcg@10': 0.847,      # Normalized Discounted Cumulative Gain
    'map@100': 0.792,      # Mean Average Precision
    'recall@100': 0.923,   # How many relevant docs in top 100
    'mrr@10': 0.851,       # Mean Reciprocal Rank
}
```

**2. Before/After Comparison:**

```python
# Load generic and fine-tuned models
generic_model = SentenceTransformer('all-MiniLM-L6-v2')
finetuned_model = SentenceTransformer('./fine_tuned_model')

# Test on your domain
def compare_models(query, candidates, relevant_ids):
    # Generic model
    generic_results = retrieve_with_model(generic_model, query, candidates)
    generic_recall = calculate_recall(generic_results, relevant_ids)

    # Fine-tuned model
    finetuned_results = retrieve_with_model(finetuned_model, query, candidates)
    finetuned_recall = calculate_recall(finetuned_results, relevant_ids)

    return {
        'generic': generic_recall,
        'finetuned': finetuned_recall,
        'improvement': finetuned_recall - generic_recall
    }

# Results example:
{
    'generic': 0.72,
    'finetuned': 0.89,
    'improvement': 0.17  # +17% recall!
}
```

---

### Q5: How do you choose the right base model for fine-tuning?

**Answer:**

**Selection criteria:**

```python
# 1. Model size vs performance trade-off
models = {
    'all-MiniLM-L6-v2': {
        'dimensions': 384,
        'params': '22M',
        'speed': '14K sentences/sec',
        'quality': 'Good',
        'use_case': 'Fast retrieval, limited resources'
    },
    'all-mpnet-base-v2': {
        'dimensions': 768,
        'params': '110M',
        'speed': '3K sentences/sec',
        'quality': 'Better',
        'use_case': 'Balanced quality/speed'
    },
    'gte-large': {
        'dimensions': 1024,
        'params': '335M',
        'speed': '1K sentences/sec',
        'quality': 'Best',
        'use_case': 'Maximum accuracy, GPU available'
    }
}

# 2. Domain alignment
# Check model's training data
# - all-MiniLM: trained on diverse web text
# - e5-base-v2: trained on retrieval tasks
# - bge-base: optimized for embeddings
```

**Practical selection:**

```python
# Start with all-MiniLM-L6-v2 if:
if data_size < 1M or latency_critical or gpu_unavailable:
    base_model = 'all-MiniLM-L6-v2'

# Use all-mpnet-base-v2 if:
elif need_better_quality and acceptable_latency:
    base_model = 'all-mpnet-base-v2'

# Use large models if:
elif gpu_available and accuracy_critical:
    base_model = 'gte-large'
```

**Test before committing:**

```python
# Quick baseline test
from sentence_transformers import SentenceTransformer

models_to_test = ['all-MiniLM-L6-v2', 'all-mpnet-base-v2', 'gte-base']
results = {}

for model_name in models_to_test:
    model = SentenceTransformer(model_name)

    # Test on your validation set
    score = evaluate_retrieval(model, val_queries, val_corpus)
    results[model_name] = score

# Choose best baseline, then fine-tune
best_base = max(results, key=results.get)
```

---

### Q6: What hyperparameters matter most in fine-tuning?

**Answer:**

**Critical hyperparameters:**

**1. Number of epochs:**

```python
# Too few: underfitting
epochs = 1
# Result: 70% accuracy (model didn't learn enough)

# Sweet spot: 2-4 epochs
epochs = 3
# Result: 87% accuracy (good learning)

# Too many: overfitting
epochs = 10
# Result: 91% train, 82% test (overfit!)

# Rule: Start with 3, use early stopping
```

**2. Batch size:**

```python
# Small batch: noisy gradients, slower
batch_size = 8
# - Training time: 120 min
# - Final accuracy: 84%

# Medium batch: good balance
batch_size = 32
# - Training time: 45 min
# - Final accuracy: 87%

# Large batch: fast but needs more data
batch_size = 128
# - Training time: 20 min
# - Final accuracy: 86% (needs more data)

# For MultipleNegativesRankingLoss: larger batch = more negatives = better
```

**3. Warmup steps:**

```python
# Calculate as 10% of total training steps
train_steps = len(train_dataloader) * epochs
warmup_steps = int(0.1 * train_steps)

# Example: 5000 samples, batch_size=32, epochs=3
# train_steps = (5000/32) * 3 = 468
# warmup_steps = 46

# Why it matters:
# - Stabilizes early training
# - Prevents large gradient updates initially
# - Typically improves final accuracy by 2-3%
```

**4. Learning rate:**

```python
# Usually auto-set by sentence-transformers (2e-5)
# But can customize:

from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=3,
    optimizer_params={'lr': 2e-5},  # Default
    # Try 1e-5 for large models, 5e-5 for small
)
```

**Complete hyperparameter template:**

```python
# For small datasets (<5K pairs)
config = {
    'epochs': 4,
    'batch_size': 16,
    'warmup_steps': 0.1,
    'learning_rate': 2e-5,
    'evaluation_steps': 200
}

# For medium datasets (5K-50K pairs)
config = {
    'epochs': 3,
    'batch_size': 32,
    'warmup_steps': 0.1,
    'learning_rate': 2e-5,
    'evaluation_steps': 500
}

# For large datasets (>50K pairs)
config = {
    'epochs': 2,
    'batch_size': 64,
    'warmup_steps': 0.1,
    'learning_rate': 1e-5,
    'evaluation_steps': 1000
}
```

---

### Q7: How do you prevent overfitting during fine-tuning?

**Answer:**

**Strategies to prevent overfitting:**

**1. Train/Dev/Test split:**

```python
from sklearn.model_selection import train_test_split

# Split properly
train, temp = train_test_split(data, test_size=0.2, random_state=42)
dev, test = train_test_split(temp, test_size=0.5, random_state=42)

# Result: 80% train, 10% dev, 10% test
```

**2. Early stopping:**

```python
from sentence_transformers import SentenceTransformer
from sentence_transformers.evaluation import InformationRetrievalEvaluator

# Setup evaluator on dev set
dev_evaluator = InformationRetrievalEvaluator(dev_queries, dev_corpus, dev_relevant)

# Track best score
best_score = 0
patience = 0
max_patience = 3

for epoch in range(10):  # Max 10 epochs
    model.fit(train_dataloader, epochs=1)

    # Evaluate on dev
    dev_score = dev_evaluator(model)

    if dev_score > best_score:
        best_score = dev_score
        model.save('best_model')
        patience = 0
    else:
        patience += 1

    # Stop if no improvement
    if patience >= max_patience:
        print(f"Early stopping at epoch {epoch}")
        break
```

**3. Data augmentation:**

```python
# Increase effective training data
def augment_training_data(examples):
    augmented = []

    for ex in examples:
        # Original
        augmented.append(ex)

        # Paraphrase
        paraphrased = paraphrase(ex.texts[0])
        augmented.append(
            InputExample(texts=[paraphrased, ex.texts[1]], label=ex.label)
        )

    return augmented

train_examples = augment_training_data(train_examples)
# Doubles training data, reduces overfitting
```

**4. Smaller learning rate:**

```python
# Smaller LR = less aggressive updates = less overfitting
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=3,
    optimizer_params={'lr': 1e-5}  # Instead of 2e-5
)
```

**5. Monitor training vs validation metrics:**

```python
# Track both during training
training_history = []

for epoch in range(epochs):
    train_loss = model.fit(train_dataloader, epochs=1)
    val_score = evaluator(model)

    training_history.append({
        'epoch': epoch,
        'train_loss': train_loss,
        'val_score': val_score
    })

    # Check for overfitting
    if len(training_history) >= 2:
        # Loss decreasing but val score not improving = overfit
        if (training_history[-1]['train_loss'] < training_history[-2]['train_loss'] and
            training_history[-1]['val_score'] < training_history[-2]['val_score']):
            print("Warning: Possible overfitting detected")
```

---

### Q8: When should you use symmetric vs asymmetric search?

**Answer:**

**Symmetric search:**
- Query and documents are same type
- Example: Document similarity, duplicate detection

**Asymmetric search:**
- Query and documents are different types
- Example: Question → Answer, Query → Document

**Implementation:**

```python
from sentence_transformers import SentenceTransformer

# Symmetric: one model for both
model = SentenceTransformer('all-MiniLM-L6-v2')

# Encode same way
doc1_emb = model.encode("Machine learning tutorial")
doc2_emb = model.encode("ML guide for beginners")
similarity = cosine_similarity([doc1_emb], [doc2_emb])

# Asymmetric: use prefix or separate encoders
model = SentenceTransformer('e5-base-v2')

# Add prefixes for asymmetric
query = "query: How to learn ML?"
doc = "passage: Machine learning is..."

query_emb = model.encode(query)
doc_emb = model.encode(doc)
```

**Fine-tuning for asymmetric:**

```python
# Training data format matters
train_examples = [
    # Asymmetric: (query, document) pairs
    InputExample(texts=['How to reset password?', 'Password reset guide: ...']),
    InputExample(texts=['What is pricing?', 'Our pricing plans: ...']),
]

# Use MultipleNegativesRankingLoss (handles asymmetry well)
train_loss = losses.MultipleNegativesRankingLoss(model)
```

**When to use which:**

```python
# Use SYMMETRIC if:
# - Comparing documents to documents
# - Finding duplicates
# - Clustering documents
tasks = ['duplicate_detection', 'document_similarity', 'clustering']

# Use ASYMMETRIC if:
# - Questions to answers
# - Queries to documents
# - Short to long text
tasks = ['QA', 'search', 'retrieval', 'RAG']
```

---

### Q9: How do you handle imbalanced training data?

**Answer:**

**Problem:**

```python
# Imbalanced categories
train_data = {
    'password_reset': 5000,  # Many examples
    'billing': 3000,
    'technical': 500,        # Few examples
    'other': 100             # Very few
}
# Model will bias toward frequent categories
```

**Solutions:**

**1. Undersample majority class:**

```python
import pandas as pd

df = pd.DataFrame(train_data)

# Undersample to match smallest class
min_count = df['category'].value_counts().min()

balanced = df.groupby('category').sample(n=min_count, random_state=42)
# Now all categories have 100 examples
```

**2. Oversample minority class:**

```python
from imblearn.over_sampling import RandomOverSampler

ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X, y)
```

**3. Weighted sampling:**

```python
from torch.utils.data import WeightedRandomSampler

# Calculate weights (inverse frequency)
class_counts = df['category'].value_counts()
weights = 1.0 / class_counts[df['category']].values

sampler = WeightedRandomSampler(
    weights=weights,
    num_samples=len(weights),
    replacement=True
)

# Use in DataLoader
train_dataloader = DataLoader(
    train_examples,
    sampler=sampler,
    batch_size=32
)
```

**4. Generate synthetic examples:**

```python
# For minority classes, generate paraphrases
def generate_synthetic_examples(examples, target_count):
    synthetic = []

    while len(examples) + len(synthetic) < target_count:
        # Pick random example
        ex = random.choice(examples)

        # Paraphrase
        paraphrased = paraphrase_model(ex.texts[0])
        synthetic.append(
            InputExample(texts=[paraphrased, ex.texts[1]], label=ex.label)
        )

    return examples + synthetic

# Balance all classes to 5000 examples each
balanced_examples = []
for category in categories:
    cat_examples = [ex for ex in train_examples if ex.category == category]
    balanced = generate_synthetic_examples(cat_examples, 5000)
    balanced_examples.extend(balanced)
```

---

### Q10: How do you fine-tune for multilingual retrieval?

**Answer:**

**Approach:**

**1. Start with multilingual base model:**

```python
from sentence_transformers import SentenceTransformer

# Multilingual models
models = [
    'paraphrase-multilingual-MiniLM-L12-v2',  # 50+ languages
    'paraphrase-multilingual-mpnet-base-v2',  # Better quality
]

model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
```

**2. Prepare multilingual training data:**

```python
# Include examples from all languages
train_examples = [
    # English
    InputExample(texts=['How to reset password?', 'Password reset guide...']),

    # Spanish
    InputExample(texts=['¿Cómo restablecer contraseña?', 'Guía de restablecimiento...']),

    # German
    InputExample(texts=['Wie setze ich das Passwort zurück?', 'Passwort zurücksetzen...']),
]

# Balance languages
language_distribution = {
    'en': 5000,
    'es': 5000,
    'de': 5000,
}
```

**3. Cross-lingual pairs:**

```python
# Train on cross-lingual similarity
train_examples = [
    # Same meaning, different languages
    InputExample(
        texts=['How to login?', 'Cómo iniciar sesión?'],
        label=1.0  # Similar
    ),
    InputExample(
        texts=['How to login?', 'Was kostet das?'],
        label=0.0  # Different
    ),
]
```

**4. Evaluation across languages:**

```python
# Test retrieval in each language
languages = ['en', 'es', 'de', 'fr']

for lang in languages:
    queries = load_queries(lang)
    corpus = load_corpus(lang)
    relevant = load_relevant(lang)

    evaluator = InformationRetrievalEvaluator(queries, corpus, relevant)
    score = evaluator(model)

    print(f"{lang}: nDCG@10 = {score}")

# Also test cross-lingual
# Query in English, retrieve from Spanish corpus
```

---

## Implementation & Coding (Q11-Q20)

### Q11: Implement complete fine-tuning pipeline

**Answer:**

```python
from sentence_transformers import (
    SentenceTransformer,
    InputExample,
    losses,
    evaluation
)
from torch.utils.data import DataLoader
import pandas as pd

class FineTuningPipeline:
    """Production fine-tuning pipeline"""

    def __init__(
        self,
        base_model: str = 'all-MiniLM-L6-v2',
        batch_size: int = 16,
        epochs: int = 3
    ):
        self.model = SentenceTransformer(base_model)
        self.batch_size = batch_size
        self.epochs = epochs

    def prepare_data(self, df: pd.DataFrame):
        """
        Prepare training data

        Args:
            df: DataFrame with columns: query, document, label (0 or 1)
        """
        train_examples = []

        for _, row in df.iterrows():
            train_examples.append(
                InputExample(
                    texts=[row['query'], row['document']],
                    label=float(row['label'])
                )
            )

        return train_examples

    def train(
        self,
        train_examples,
        evaluator=None,
        output_path='./fine_tuned_model'
    ):
        """
        Train model

        Args:
            train_examples: List of InputExample
            evaluator: Optional evaluator for validation
            output_path: Where to save model
        """
        # Create dataloader
        train_dataloader = DataLoader(
            train_examples,
            shuffle=True,
            batch_size=self.batch_size
        )

        # Use MultipleNegativesRankingLoss for retrieval
        train_loss = losses.MultipleNegativesRankingLoss(self.model)

        # Calculate warmup steps (10% of total)
        total_steps = len(train_dataloader) * self.epochs
        warmup_steps = int(0.1 * total_steps)

        # Train
        self.model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=self.epochs,
            warmup_steps=warmup_steps,
            evaluator=evaluator,
            evaluation_steps=500,
            output_path=output_path,
            save_best_model=True,
            show_progress_bar=True
        )

        return self.model

# Usage
if __name__ == "__main__":
    # Load data
    df = pd.read_csv('training_data.csv')

    # Initialize pipeline
    pipeline = FineTuningPipeline(
        base_model='all-MiniLM-L6-v2',
        batch_size=16,
        epochs=3
    )

    # Prepare data
    train_examples = pipeline.prepare_data(df)

    # Train
    model = pipeline.train(
        train_examples,
        output_path='./fine_tuned_model'
    )

    print("Training complete!")
```

---

### Q12: How do you create training data from support tickets?

**Answer:**

```python
import pandas as pd
from sentence_transformers import InputExample
import random

def create_training_from_tickets(tickets_df):
    """
    Create training pairs from categorized tickets

    Args:
        tickets_df: DataFrame with columns [text, category]

    Returns:
        List of InputExample for training
    """
    examples = []

    # Group by category
    categories = tickets_df.groupby('category')

    # Positive pairs: same category
    for category, group in categories:
        texts = group['text'].tolist()

        # Create pairs within category
        for i in range(len(texts)):
            for j in range(i + 1, min(i + 5, len(texts))):  # Limit pairs
                examples.append(
                    InputExample(
                        texts=[texts[i], texts[j]],
                        label=1.0  # Similar
                    )
                )

    # Negative pairs: different categories
    categories_list = list(categories.groups.keys())

    for _ in range(len(examples)):  # Balance pos/neg
        cat1, cat2 = random.sample(categories_list, 2)

        text1 = random.choice(tickets_df[tickets_df['category'] == cat1]['text'].tolist())
        text2 = random.choice(tickets_df[tickets_df['category'] == cat2]['text'].tolist())

        examples.append(
            InputExample(
                texts=[text1, text2],
                label=0.0  # Different
            )
        )

    random.shuffle(examples)
    return examples

# Usage
tickets = pd.read_csv('support_tickets.csv')
train_examples = create_training_from_tickets(tickets)
print(f"Created {len(train_examples)} training pairs")
```

---

### Q13: Implement training data from Q&A pairs

**Answer:**

```python
import pandas as pd
from sentence_transformers import InputExample
import random

def create_qa_training_data(qa_df):
    """
    Create training data from Q&A pairs

    Args:
        qa_df: DataFrame with columns [question, answer]

    Returns:
        List of InputExample
    """
    examples = []
    questions = qa_df['question'].tolist()
    answers = qa_df['answer'].tolist()

    for i, (question, answer) in enumerate(zip(questions, answers)):
        # Positive: question -> correct answer
        examples.append(
            InputExample(texts=[question, answer])
        )

        # Hard negative: question -> random answer (not too easy)
        # Sample from similar category if available
        negative_indices = [j for j in range(len(answers)) if j != i]
        neg_idx = random.choice(negative_indices)

        examples.append(
            InputExample(texts=[question, answers[neg_idx]])
        )

    return examples

# For MultipleNegativesRankingLoss (no explicit negatives needed)
def create_qa_for_mnrl(qa_df):
    """Only positive pairs - negatives come from batch"""
    examples = []

    for _, row in qa_df.iterrows():
        examples.append(
            InputExample(texts=[row['question'], row['answer']])
        )

    return examples

# Usage
qa_pairs = pd.read_csv('qa_pairs.csv')
train_examples = create_qa_for_mnrl(qa_pairs)
```

---

### Q14: How do you implement custom evaluation during training?

**Answer:**

```python
from sentence_transformers.evaluation import SentenceEvaluator
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class CustomRetrievalEvaluator(SentenceEvaluator):
    """Custom evaluator for domain-specific retrieval"""

    def __init__(self, queries, corpus, relevant_docs, name=''):
        self.queries = queries
        self.corpus = corpus
        self.relevant_docs = relevant_docs
        self.name = name

    def __call__(self, model, output_path=None, epoch=-1, steps=-1):
        """
        Evaluate model

        Returns:
            float: recall@10 score
        """
        # Encode queries and corpus
        query_ids = list(self.queries.keys())
        query_texts = [self.queries[qid] for qid in query_ids]
        query_embeddings = model.encode(query_texts, convert_to_numpy=True)

        corpus_ids = list(self.corpus.keys())
        corpus_texts = [self.corpus[cid] for cid in corpus_ids]
        corpus_embeddings = model.encode(corpus_texts, convert_to_numpy=True)

        # Calculate similarities
        similarities = cosine_similarity(query_embeddings, corpus_embeddings)

        # Calculate recall@10
        recalls = []
        for i, qid in enumerate(query_ids):
            # Get top 10 documents
            top_10_indices = np.argsort(similarities[i])[-10:][::-1]
            top_10_ids = [corpus_ids[idx] for idx in top_10_indices]

            # Check how many relevant docs in top 10
            relevant = self.relevant_docs[qid]
            found = len(set(top_10_ids) & set(relevant))
            recall = found / len(relevant) if relevant else 0
            recalls.append(recall)

        avg_recall = np.mean(recalls)

        print(f"Recall@10: {avg_recall:.4f}")
        return avg_recall

# Usage in training
evaluator = CustomRetrievalEvaluator(
    queries=dev_queries,
    corpus=dev_corpus,
    relevant_docs=dev_relevant
)

model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=3,
    evaluator=evaluator,
    evaluation_steps=500,  # Evaluate every 500 steps
    output_path='./model'
)
```

---

### Q15: Implement data augmentation for fine-tuning

**Answer:**

```python
import random
import nlpaug.augmenter.word as naw
from sentence_transformers import InputExample

class DataAugmenter:
    """Augment training data for fine-tuning"""

    def __init__(self):
        # Synonym replacement
        self.syn_aug = naw.SynonymAug(aug_src='wordnet')

        # Back-translation (requires model)
        # self.back_trans = naw.BackTranslationAug(
        #     from_model_name='facebook/wmt19-en-de',
        #     to_model_name='facebook/wmt19-de-en'
        # )

    def synonym_replacement(self, text, aug_p=0.3):
        """Replace words with synonyms"""
        return self.syn_aug.augment(text, n=1)[0]

    def random_deletion(self, text, p=0.1):
        """Randomly delete words"""
        words = text.split()
        if len(words) == 1:
            return text

        new_words = []
        for word in words:
            if random.random() > p:
                new_words.append(word)

        if len(new_words) == 0:
            return random.choice(words)

        return ' '.join(new_words)

    def random_swap(self, text, n=1):
        """Randomly swap words"""
        words = text.split()
        if len(words) < 2:
            return text

        for _ in range(n):
            idx1, idx2 = random.sample(range(len(words)), 2)
            words[idx1], words[idx2] = words[idx2], words[idx1]

        return ' '.join(words)

    def augment_examples(self, examples, augment_ratio=1.0):
        """
        Augment training examples

        Args:
            examples: List of InputExample
            augment_ratio: How many augmented examples to create (1.0 = double data)

        Returns:
            List with original + augmented examples
        """
        augmented = list(examples)  # Copy originals
        num_to_augment = int(len(examples) * augment_ratio)

        for _ in range(num_to_augment):
            # Pick random example
            ex = random.choice(examples)

            # Augment first text (query)
            aug_method = random.choice([
                self.synonym_replacement,
                self.random_deletion,
                self.random_swap
            ])

            aug_text = aug_method(ex.texts[0])

            # Create new example
            augmented.append(
                InputExample(
                    texts=[aug_text, ex.texts[1]],
                    label=ex.label if hasattr(ex, 'label') else None
                )
            )

        random.shuffle(augmented)
        return augmented

# Usage
augmenter = DataAugmenter()

# Original: 1000 examples
original_examples = load_training_data()

# Augmented: 2000 examples (1000 original + 1000 augmented)
augmented_examples = augmenter.augment_examples(original_examples, augment_ratio=1.0)

print(f"Original: {len(original_examples)}")
print(f"Augmented: {len(augmented_examples)}")
```

---

### Q16: How do you mine hard negatives for training?

**Answer:**

```python
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def mine_hard_negatives(model, queries, corpus, relevant_docs, top_k=50):
    """
    Mine hard negatives: documents that are similar but not relevant

    Args:
        model: SentenceTransformer model
        queries: Dict of {query_id: query_text}
        corpus: Dict of {doc_id: doc_text}
        relevant_docs: Dict of {query_id: set of relevant doc_ids}
        top_k: How many top documents to consider

    Returns:
        Dict of {query_id: list of hard negative doc_ids}
    """
    # Encode all
    query_ids = list(queries.keys())
    query_texts = [queries[qid] for qid in query_ids]
    query_embs = model.encode(query_texts)

    corpus_ids = list(corpus.keys())
    corpus_texts = [corpus[cid] for cid in corpus_ids]
    corpus_embs = model.encode(corpus_texts)

    # Calculate similarities
    sims = cosine_similarity(query_embs, corpus_embs)

    hard_negatives = {}

    for i, qid in enumerate(query_ids):
        # Get top-k most similar documents
        top_indices = np.argsort(sims[i])[-top_k:][::-1]
        top_doc_ids = [corpus_ids[idx] for idx in top_indices]

        # Filter out relevant docs (those are positives)
        relevant = relevant_docs.get(qid, set())

        # Hard negatives: high similarity but not relevant
        hard_negs = [
            doc_id for doc_id in top_doc_ids
            if doc_id not in relevant
        ][:10]  # Keep top 10 hard negatives

        hard_negatives[qid] = hard_negs

    return hard_negatives

# Create training data with hard negatives
def create_training_with_hard_negatives(queries, corpus, relevant_docs, hard_negatives):
    """Create training examples using mined hard negatives"""
    examples = []

    for qid, query_text in queries.items():
        # Positive examples
        for doc_id in relevant_docs[qid]:
            examples.append(
                InputExample(
                    texts=[query_text, corpus[doc_id]],
                    label=1.0
                )
            )

        # Hard negative examples
        for doc_id in hard_negatives.get(qid, []):
            examples.append(
                InputExample(
                    texts=[query_text, corpus[doc_id]],
                    label=0.0
                )
            )

    return examples

# Usage
base_model = SentenceTransformer('all-MiniLM-L6-v2')

hard_negs = mine_hard_negatives(
    base_model,
    train_queries,
    train_corpus,
    train_relevant_docs
)

train_examples = create_training_with_hard_negatives(
    train_queries,
    train_corpus,
    train_relevant_docs,
    hard_negs
)
```

---

### Q17: Implement model comparison and A/B testing

**Answer:**

```python
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

class ModelComparison:
    """Compare multiple models for retrieval"""

    def __init__(self, queries, corpus, relevant_docs):
        self.queries = queries
        self.corpus = corpus
        self.relevant_docs = relevant_docs

    def recall_at_k(self, similarities, k=10):
        """Calculate recall@k for all queries"""
        query_ids = list(self.queries.keys())
        corpus_ids = list(self.corpus.keys())

        recalls = []
        for i, qid in enumerate(query_ids):
            # Get top-k
            top_k_indices = np.argsort(similarities[i])[-k:][::-1]
            top_k_ids = [corpus_ids[idx] for idx in top_k_indices]

            # Calculate recall
            relevant = self.relevant_docs[qid]
            found = len(set(top_k_ids) & set(relevant))
            recall = found / len(relevant) if relevant else 0
            recalls.append(recall)

        return np.mean(recalls)

    def mrr(self, similarities):
        """Calculate Mean Reciprocal Rank"""
        query_ids = list(self.queries.keys())
        corpus_ids = list(self.corpus.keys())

        reciprocal_ranks = []
        for i, qid in enumerate(query_ids):
            # Get ranked documents
            ranked_indices = np.argsort(similarities[i])[::-1]
            ranked_ids = [corpus_ids[idx] for idx in ranked_indices]

            # Find first relevant document
            relevant = self.relevant_docs[qid]
            for rank, doc_id in enumerate(ranked_ids, 1):
                if doc_id in relevant:
                    reciprocal_ranks.append(1.0 / rank)
                    break
            else:
                reciprocal_ranks.append(0.0)

        return np.mean(reciprocal_ranks)

    def evaluate_model(self, model):
        """Evaluate single model"""
        # Encode
        query_texts = [self.queries[qid] for qid in self.queries.keys()]
        corpus_texts = [self.corpus[cid] for cid in self.corpus.keys()]

        query_embs = model.encode(query_texts)
        corpus_embs = model.encode(corpus_texts)

        # Similarities
        sims = cosine_similarity(query_embs, corpus_embs)

        # Metrics
        return {
            'recall@10': self.recall_at_k(sims, k=10),
            'recall@100': self.recall_at_k(sims, k=100),
            'mrr': self.mrr(sims)
        }

    def compare_models(self, models_dict):
        """
        Compare multiple models

        Args:
            models_dict: {model_name: model_path_or_object}

        Returns:
            DataFrame with comparison results
        """
        results = []

        for name, model_path in models_dict.items():
            print(f"Evaluating {name}...")
            model = SentenceTransformer(model_path) if isinstance(model_path, str) else model_path

            metrics = self.evaluate_model(model)
            metrics['model'] = name
            results.append(metrics)

        df = pd.DataFrame(results)
        return df.sort_values('recall@10', ascending=False)

# Usage
comparison = ModelComparison(test_queries, test_corpus, test_relevant_docs)

models = {
    'generic': 'all-MiniLM-L6-v2',
    'finetuned_v1': './fine_tuned_model',
    'finetuned_v2': './fine_tuned_model_v2',
}

results_df = comparison.compare_models(models)
print(results_df)

# Output:
#         model  recall@10  recall@100      mrr
# 1  finetuned_v2      0.89        0.96    0.84
# 0  finetuned_v1      0.87        0.94    0.81
# 2       generic      0.72        0.88    0.68
```

---

### Q18: How do you implement incremental training?

**Answer:**

```python
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import os

class IncrementalTrainer:
    """Incrementally update model with new data"""

    def __init__(self, model_path):
        """Load existing model or create new one"""
        if os.path.exists(model_path):
            print(f"Loading existing model from {model_path}")
            self.model = SentenceTransformer(model_path)
        else:
            print("Creating new model")
            self.model = SentenceTransformer('all-MiniLM-L6-v2')

        self.model_path = model_path

    def train_increment(self, new_examples, epochs=1, batch_size=16):
        """
        Train on new data without forgetting old

        Args:
            new_examples: New training examples
            epochs: Number of epochs (use fewer for incremental)
            batch_size: Batch size
        """
        print(f"Training on {len(new_examples)} new examples")

        # Create dataloader
        dataloader = DataLoader(new_examples, shuffle=True, batch_size=batch_size)

        # Use lower learning rate for incremental training
        train_loss = losses.MultipleNegativesRankingLoss(self.model)

        # Train with smaller LR to avoid catastrophic forgetting
        self.model.fit(
            train_objectives=[(dataloader, train_loss)],
            epochs=epochs,
            warmup_steps=int(0.1 * len(dataloader)),
            optimizer_params={'lr': 1e-5},  # Lower LR
            output_path=self.model_path,
            show_progress_bar=True
        )

        print(f"Model saved to {self.model_path}")

    def train_with_replay(self, new_examples, old_examples, epochs=1):
        """
        Train with experience replay to prevent forgetting

        Args:
            new_examples: New training data
            old_examples: Sample of old training data
            epochs: Training epochs
        """
        # Mix new and old data (50/50 split)
        mixed_examples = new_examples + old_examples[:len(new_examples)]

        print(f"Training on {len(mixed_examples)} examples (new + old)")
        self.train_increment(mixed_examples, epochs=epochs)

# Usage scenario
trainer = IncrementalTrainer('./my_model')

# Week 1: Initial training
week1_data = load_data('week1')
trainer.train_increment(week1_data, epochs=3)

# Week 2: New data arrives
week2_data = load_data('week2')
trainer.train_with_replay(
    new_examples=week2_data,
    old_examples=week1_data,  # Replay old data
    epochs=1
)

# Week 3: More new data
week3_data = load_data('week3')
trainer.train_with_replay(
    new_examples=week3_data,
    old_examples=week1_data + week2_data,
    epochs=1
)
```

---

### Q19: Implement distributed fine-tuning

**Answer:**

```python
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import os

def setup_distributed():
    """Initialize distributed training"""
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    return local_rank

def cleanup_distributed():
    """Cleanup distributed training"""
    dist.destroy_process_group()

class DistributedFineTuner:
    """Distributed fine-tuning across multiple GPUs"""

    def __init__(self, base_model='all-MiniLM-L6-v2'):
        self.local_rank = setup_distributed()
        self.model = SentenceTransformer(base_model, device=f'cuda:{self.local_rank}')

        # Wrap with DDP
        # Note: sentence-transformers handles this internally in newer versions

    def train(self, train_examples, epochs=3, batch_size=32):
        """
        Train model in distributed manner

        Args:
            train_examples: Training data
            epochs: Number of epochs
            batch_size: Batch size per GPU
        """
        # Create dataloader with distributed sampler
        from torch.utils.data.distributed import DistributedSampler

        sampler = DistributedSampler(
            train_examples,
            num_replicas=dist.get_world_size(),
            rank=dist.get_rank()
        )

        dataloader = DataLoader(
            train_examples,
            batch_size=batch_size,
            sampler=sampler
        )

        # Training loss
        train_loss = losses.MultipleNegativesRankingLoss(self.model)

        # Train
        self.model.fit(
            train_objectives=[(dataloader, train_loss)],
            epochs=epochs,
            warmup_steps=int(0.1 * len(dataloader)),
            output_path='./distributed_model' if self.local_rank == 0 else None,
            show_progress_bar=(self.local_rank == 0)
        )

        # Only rank 0 saves
        if self.local_rank == 0:
            self.model.save('./distributed_model')

        cleanup_distributed()

# Launch script (save as train_distributed.py)
"""
# Run with torchrun
torchrun --nproc_per_node=4 train_distributed.py

# Or with python -m torch.distributed.launch
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    train_distributed.py
"""

# Simple alternative: DataParallel (single machine, multiple GPUs)
class SimpleMultiGPU:
    """Simple multi-GPU training with DataParallel"""

    def __init__(self, base_model='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(base_model)

        # Check available GPUs
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs")
            # sentence-transformers handles multi-GPU internally
            self.model = self.model.to('cuda')

    def train(self, train_examples, epochs=3, batch_size=32):
        """Train with automatic multi-GPU"""
        dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
        train_loss = losses.MultipleNegativesRankingLoss(self.model)

        self.model.fit(
            train_objectives=[(dataloader, train_loss)],
            epochs=epochs,
            output_path='./multigpu_model'
        )

# Usage
if __name__ == '__main__':
    # Load data
    train_examples = load_training_data()

    # Simple multi-GPU
    trainer = SimpleMultiGPU()
    trainer.train(train_examples)
```

---

### Q20: How do you version and track fine-tuned models?

**Answer:**

```python
import json
import os
from datetime import datetime
from sentence_transformers import SentenceTransformer
import hashlib

class ModelVersioning:
    """Track and version fine-tuned models"""

    def __init__(self, base_path='./models'):
        self.base_path = base_path
        os.makedirs(base_path, exist_ok=True)

    def save_model_with_metadata(
        self,
        model,
        version_name,
        training_config,
        metrics,
        training_data_info
    ):
        """
        Save model with complete metadata

        Args:
            model: SentenceTransformer model
            version_name: Version identifier
            training_config: Dict with hyperparameters
            metrics: Dict with evaluation metrics
            training_data_info: Info about training data
        """
        # Create version directory
        model_dir = os.path.join(self.base_path, version_name)
        os.makedirs(model_dir, exist_ok=True)

        # Save model
        model.save(model_dir)

        # Create metadata
        metadata = {
            'version': version_name,
            'timestamp': datetime.now().isoformat(),
            'base_model': training_config.get('base_model'),
            'training_config': training_config,
            'metrics': metrics,
            'training_data': training_data_info,
            'model_hash': self._compute_model_hash(model_dir)
        }

        # Save metadata
        with open(os.path.join(model_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)

        # Update registry
        self._update_registry(metadata)

        print(f"Model saved: {model_dir}")
        return model_dir

    def _compute_model_hash(self, model_dir):
        """Compute hash of model files for integrity"""
        hasher = hashlib.sha256()

        for file in os.listdir(model_dir):
            if file.endswith('.bin') or file.endswith('.safetensors'):
                filepath = os.path.join(model_dir, file)
                with open(filepath, 'rb') as f:
                    hasher.update(f.read())

        return hasher.hexdigest()

    def _update_registry(self, metadata):
        """Update central registry of all models"""
        registry_path = os.path.join(self.base_path, 'registry.json')

        # Load existing registry
        if os.path.exists(registry_path):
            with open(registry_path, 'r') as f:
                registry = json.load(f)
        else:
            registry = {'models': []}

        # Add new model
        registry['models'].append({
            'version': metadata['version'],
            'timestamp': metadata['timestamp'],
            'metrics': metadata['metrics'],
            'path': os.path.join(self.base_path, metadata['version'])
        })

        # Save registry
        with open(registry_path, 'w') as f:
            json.dump(registry, f, indent=2)

    def list_models(self):
        """List all tracked models"""
        registry_path = os.path.join(self.base_path, 'registry.json')

        if not os.path.exists(registry_path):
            return []

        with open(registry_path, 'r') as f:
            registry = json.load(f)

        return registry['models']

    def load_model(self, version_name):
        """Load specific model version"""
        model_dir = os.path.join(self.base_path, version_name)

        if not os.path.exists(model_dir):
            raise ValueError(f"Model version {version_name} not found")

        # Load metadata
        with open(os.path.join(model_dir, 'metadata.json'), 'r') as f:
            metadata = json.load(f)

        # Load model
        model = SentenceTransformer(model_dir)

        return model, metadata

    def get_best_model(self, metric='recall@10'):
        """Get best model by metric"""
        models = self.list_models()

        if not models:
            return None

        best = max(models, key=lambda m: m['metrics'].get(metric, 0))
        return self.load_model(best['version'])

# Usage
versioning = ModelVersioning('./models')

# After training
model = train_model()

# Save with metadata
versioning.save_model_with_metadata(
    model=model,
    version_name='v1.2.0_medical',
    training_config={
        'base_model': 'all-MiniLM-L6-v2',
        'epochs': 3,
        'batch_size': 32,
        'loss': 'MultipleNegativesRankingLoss'
    },
    metrics={
        'recall@10': 0.89,
        'mrr': 0.84,
        'ndcg@10': 0.87
    },
    training_data_info={
        'num_examples': 10000,
        'source': 'medical_qa_pairs',
        'date_range': '2024-01-01 to 2024-12-31'
    }
)

# List all models
models = versioning.list_models()
for m in models:
    print(f"{m['version']}: recall@10 = {m['metrics']['recall@10']}")

# Load best model
best_model, metadata = versioning.get_best_model('recall@10')
```

---

## Debugging & Troubleshooting (Q21-Q25)

### Q21: Model performance degrades after fine-tuning - how to debug?

**Answer:**

**Diagnostic checklist:**

**1. Compare before/after:**

```python
# Evaluate both models on same test set
base_model = SentenceTransformer('all-MiniLM-L6-v2')
finetuned_model = SentenceTransformer('./fine_tuned_model')

def evaluate_model(model, test_queries, test_corpus, test_relevant):
    # ... evaluation code ...
    return metrics

base_metrics = evaluate_model(base_model, test_queries, test_corpus, test_relevant)
ft_metrics = evaluate_model(finetuned_model, test_queries, test_corpus, test_relevant)

print("Base model:", base_metrics)
print("Fine-tuned:", ft_metrics)

# If fine-tuned is worse: overfitting or wrong loss function
```

**2. Check for data leakage:**

```python
# Are test examples in training data?
def check_leakage(train_examples, test_queries):
    train_texts = set()
    for ex in train_examples:
        train_texts.add(ex.texts[0])
        train_texts.add(ex.texts[1])

    leakage = []
    for qid, query in test_queries.items():
        if query in train_texts:
            leakage.append(qid)

    if leakage:
        print(f"WARNING: {len(leakage)} test queries in training data!")
    return leakage
```

**3. Analyze failure cases:**

```python
# Find queries where fine-tuned model fails
def find_failures(base_model, ft_model, test_queries, test_corpus, test_relevant):
    failures = []

    for qid, query in test_queries.items():
        # Get top result from each model
        base_top = retrieve_top1(base_model, query, test_corpus)
        ft_top = retrieve_top1(ft_model, query, test_corpus)

        relevant = test_relevant[qid]

        # Cases where base was right but FT is wrong
        if base_top in relevant and ft_top not in relevant:
            failures.append({
                'query': query,
                'base_retrieved': base_top,
                'ft_retrieved': ft_top,
                'relevant': relevant
            })

    return failures

# Analyze patterns in failures
failures = find_failures(base_model, finetuned_model, test_queries, test_corpus, test_relevant)
print(f"Found {len(failures)} regression cases")

# Look for patterns
for f in failures[:5]:
    print(f"Query: {f['query']}")
    print(f"FT retrieved (wrong): {f['ft_retrieved']}")
    print(f"Should retrieve: {f['relevant']}")
    print()
```

**4. Check training curves:**

```python
# Training loss should decrease
# If loss stops decreasing: learning rate too low or converged
# If loss fluctuates: batch size too small or LR too high

# Monitor validation score during training
# If val score decreases while train improves: overfitting
```

**Common causes and fixes:**

```python
# Issue 1: Overfitting
# Symptoms: Great train, poor test performance
# Fix: Early stopping, more data, data augmentation
if train_accuracy > 0.95 and test_accuracy < 0.75:
    print("Overfitting detected")
    # Solutions:
    # - Reduce epochs
    # - Add more training data
    # - Use data augmentation

# Issue 2: Wrong loss function
# Symptoms: Model performs poorly on asymmetric tasks
# Fix: Use MultipleNegativesRankingLoss for retrieval
if task == 'retrieval' and loss_function == 'ContrastiveLoss':
    print("Consider using MultipleNegativesRankingLoss")

# Issue 3: Imbalanced data
# Symptoms: Good on frequent categories, poor on rare ones
# Fix: Balance training data
category_counts = count_categories(train_data)
if max(category_counts.values()) / min(category_counts.values()) > 10:
    print("Highly imbalanced data - consider balancing")

# Issue 4: Domain shift
# Symptoms: Good on train domain, poor on test domain
# Fix: Include more diverse examples in training
```

---

### Q22: Training is too slow - how to speed up?

**Answer:**

**Optimization strategies:**

**1. Increase batch size:**

```python
# Small batch: slow
batch_size = 8
# Training time: 120 minutes

# Larger batch: faster (if GPU memory allows)
batch_size = 64
# Training time: 30 minutes

# Check GPU memory usage
import torch
print(f"GPU memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

# Find max batch size
for batch_size in [16, 32, 64, 128]:
    try:
        dataloader = DataLoader(train_examples, batch_size=batch_size)
        # Try one batch
        model.fit(train_objectives=[(dataloader, train_loss)], epochs=1)
        print(f"Batch size {batch_size}: OK")
    except RuntimeError as e:
        if 'out of memory' in str(e):
            print(f"Batch size {batch_size}: Too large")
            break
```

**2. Use mixed precision training:**

```python
# Automatically handled by newer sentence-transformers
# Uses FP16 instead of FP32 for faster computation

from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

# Mixed precision is often automatic on newer GPUs
# Explicitly enable:
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=3,
    use_amp=True  # Automatic Mixed Precision
)
```

**3. Reduce sequence length:**

```python
# Longer sequences = slower
# Truncate if possible

# Check actual lengths
lengths = [len(ex.texts[0].split()) for ex in train_examples]
print(f"Mean length: {np.mean(lengths)}")
print(f"95th percentile: {np.percentile(lengths, 95)}")

# If most are short, set max_seq_length
model = SentenceTransformer('all-MiniLM-L6-v2')
model.max_seq_length = 128  # Default is 512

# Speed improvement: 2-4x for short texts
```

**4. Use smaller base model:**

```python
# all-mpnet-base-v2: slow but accurate
# all-MiniLM-L6-v2: fast but slightly less accurate

models_speed = {
    'all-MiniLM-L6-v2': '14K sentences/sec',  # Fastest
    'all-MiniLM-L12-v2': '9K sentences/sec',
    'all-mpnet-base-v2': '3K sentences/sec',   # Slowest
}

# Start with MiniLM, upgrade if needed
```

**5. Reduce training data (carefully):**

```python
# If you have 100K examples, maybe 10K is enough?
# Test on subset first

def sample_training_data(examples, sample_size):
    """Sample diverse subset of training data"""
    # Stratified sampling if categorical labels
    # Random sampling otherwise
    return random.sample(examples, sample_size)

# Train on subset
subset = sample_training_data(train_examples, 10000)
# Much faster training, similar results if data is redundant
```

**6. Use gradient accumulation:**

```python
# Simulate large batch with multiple small batches
# Useful when GPU memory is limited

# Instead of batch_size=128 (OOM)
# Use batch_size=32 with accumulation_steps=4

# This is handled automatically by adjusting batch_size and epochs
# Effective batch size = batch_size * accumulation_steps
```

**Benchmarking:**

```python
import time

def benchmark_training(config):
    model = SentenceTransformer(config['base_model'])

    dataloader = DataLoader(
        train_examples,
        batch_size=config['batch_size'],
        shuffle=True
    )

    train_loss = losses.MultipleNegativesRankingLoss(model)

    start = time.time()
    model.fit(
        train_objectives=[(dataloader, train_loss)],
        epochs=1
    )
    elapsed = time.time() - start

    return elapsed

# Test different configs
configs = [
    {'base_model': 'all-MiniLM-L6-v2', 'batch_size': 32},
    {'base_model': 'all-MiniLM-L6-v2', 'batch_size': 64},
    {'base_model': 'all-mpnet-base-v2', 'batch_size': 32},
]

for config in configs:
    time_taken = benchmark_training(config)
    print(f"{config}: {time_taken:.1f} seconds")
```

---

### Q23: How do you handle out-of-memory errors?

**Answer:**

**Solutions:**

**1. Reduce batch size:**

```python
# Start large and reduce until it fits
for batch_size in [128, 64, 32, 16, 8]:
    try:
        dataloader = DataLoader(train_examples, batch_size=batch_size)
        model.fit(train_objectives=[(dataloader, train_loss)], epochs=1)
        print(f"✓ Batch size {batch_size} works")
        break
    except RuntimeError as e:
        if 'out of memory' in str(e):
            print(f"✗ Batch size {batch_size} too large")
            torch.cuda.empty_cache()  # Clear cache
        else:
            raise e
```

**2. Gradient accumulation:**

```python
# Simulate large batch with small batches
# sentence-transformers doesn't directly support this
# But you can achieve similar effect:

# Instead of batch_size=128, epochs=3
# Use batch_size=32, epochs=12 (same total updates)

# Effective training is similar
```

**3. Clear GPU cache:**

```python
import torch

# Between training runs
torch.cuda.empty_cache()

# Force garbage collection
import gc
gc.collect()
torch.cuda.empty_cache()
```

**4. Use CPU offloading:**

```python
# Move some layers to CPU (slower but works)
# Not directly supported in sentence-transformers
# Use gradient checkpointing instead

# For very large models
model = SentenceTransformer('large-model')

# Reduce memory by recomputing activations
# Automatically handled by PyTorch for some models
```

**5. Process in chunks:**

```python
# For encoding large corpus
def encode_in_chunks(model, texts, batch_size=32, max_chunk_size=10000):
    """Encode large dataset in chunks"""
    all_embeddings = []

    for i in range(0, len(texts), max_chunk_size):
        chunk = texts[i:i+max_chunk_size]

        # Encode chunk
        embeddings = model.encode(
            chunk,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )

        all_embeddings.append(embeddings)

        # Clear cache
        torch.cuda.empty_cache()

    return np.vstack(all_embeddings)

# Usage
large_corpus = load_large_corpus()  # 1M documents
embeddings = encode_in_chunks(model, large_corpus)
```

**6. Use smaller model:**

```python
# If all else fails, use smaller base model
# all-MiniLM-L6-v2 (22M params) vs all-mpnet-base-v2 (110M params)

# MiniLM uses 5x less memory
model = SentenceTransformer('all-MiniLM-L6-v2')
```

**7. Monitor memory usage:**

```python
import torch

def print_gpu_memory():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        print(f"GPU Memory: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")

# Check before training
print_gpu_memory()

# Check during training
# Add to custom callback or evaluator

# Find memory leaks
for i in range(5):
    embeddings = model.encode(sample_texts)
    print(f"Iteration {i+1}:")
    print_gpu_memory()
    del embeddings  # Explicit deletion
    torch.cuda.empty_cache()
```

---

### Q24: Fine-tuned model gives inconsistent results - how to debug?

**Answer:**

**Causes and solutions:**

**1. Random initialization not fixed:**

```python
# Set all random seeds
import random
import numpy as np
import torch

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Deterministic behavior (slower)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# Now training is reproducible
```

**2. Batch order varies:**

```python
# Use fixed seed in DataLoader
dataloader = DataLoader(
    train_examples,
    shuffle=True,
    batch_size=32,
    generator=torch.Generator().manual_seed(42)  # Fixed shuffle
)
```

**3. Embeddings not normalized:**

```python
# Always normalize for consistent similarity scores
embeddings = model.encode(texts, normalize_embeddings=True)

# Check normalization
norms = np.linalg.norm(embeddings, axis=1)
print(f"Embedding norms: min={norms.min():.4f}, max={norms.max():.4f}")
# Should be close to 1.0 if normalized
```

**4. Test different queries give different results:**

```python
# Debug similarity scores
def debug_retrieval(model, query, corpus):
    query_emb = model.encode(query, normalize_embeddings=True)
    corpus_embs = model.encode(corpus, normalize_embeddings=True)

    similarities = cosine_similarity([query_emb], corpus_embs)[0]

    # Print top results with scores
    top_indices = np.argsort(similarities)[-5:][::-1]

    print(f"Query: {query}")
    for idx in top_indices:
        print(f"  {similarities[idx]:.4f}: {corpus[idx]}")
    print()

# Test on same query multiple times
for i in range(3):
    debug_retrieval(model, "How to reset password?", corpus)

# If results vary: model is not deterministic or embeddings not cached
```

**5. Model updates between calls:**

```python
# Make sure model is in eval mode
model.eval()  # Disable dropout

# Don't fine-tune during inference
# Load model once, use many times
```

**6. Floating point precision:**

```python
# Different hardware can give slightly different results
# Use consistent precision

# FP32 vs FP16 can differ slightly
embeddings_fp32 = model.encode(texts, precision='float32')
embeddings_fp16 = model.encode(texts, precision='float16')

# Check difference
diff = np.abs(embeddings_fp32 - embeddings_fp16).max()
print(f"Max difference: {diff}")
# Should be very small (<0.01) but not zero
```

**7. Cache embeddings for consistency:**

```python
import joblib

# Compute once, save
embeddings = model.encode(corpus, normalize_embeddings=True)
joblib.dump(embeddings, 'corpus_embeddings.pkl')

# Load for subsequent runs
embeddings = joblib.load('corpus_embeddings.pkl')

# Now retrieval is perfectly consistent
```

---

### Q25: How do you debug training when loss doesn't decrease?

**Answer:**

**Diagnostic steps:**

**1. Check loss value:**

```python
# Is loss actually not decreasing, or just fluctuating?
# Plot training loss

import matplotlib.pyplot as plt

losses = []

for epoch in range(epochs):
    epoch_losses = []

    for batch in train_dataloader:
        loss = train_step(batch)  # Your training step
        epoch_losses.append(loss)

    avg_loss = np.mean(epoch_losses)
    losses.append(avg_loss)
    print(f"Epoch {epoch}: Loss = {avg_loss:.4f}")

plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.show()
```

**2. Check learning rate:**

```python
# Learning rate too low: loss decreases very slowly
# Learning rate too high: loss fluctuates or doesn't decrease

# Try different learning rates
learning_rates = [1e-6, 5e-6, 1e-5, 2e-5, 5e-5, 1e-4]

for lr in learning_rates:
    model = SentenceTransformer('all-MiniLM-L6-v2')

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=1,
        optimizer_params={'lr': lr}
    )

    # Evaluate
    loss = evaluate_loss(model, val_dataloader)
    print(f"LR {lr}: Loss = {loss:.4f}")

# Find best LR
```

**3. Check data quality:**

```python
# Bad data: model can't learn
def check_data_quality(examples):
    """Sanity check training data"""

    # Check for empty texts
    empty = sum(1 for ex in examples if not ex.texts[0] or not ex.texts[1])
    if empty > 0:
        print(f"WARNING: {empty} examples with empty text")

    # Check label distribution
    if hasattr(examples[0], 'label'):
        labels = [ex.label for ex in examples]
        print(f"Label distribution: {np.bincount([int(l) for l in labels])}")

    # Check for duplicates
    text_pairs = [(ex.texts[0], ex.texts[1]) for ex in examples]
    unique = len(set(text_pairs))
    print(f"Unique pairs: {unique}/{len(examples)}")

    # Sample some examples
    print("\nSample examples:")
    for ex in examples[:3]:
        print(f"  Text1: {ex.texts[0][:50]}...")
        print(f"  Text2: {ex.texts[1][:50]}...")
        if hasattr(ex, 'label'):
            print(f"  Label: {ex.label}")
        print()

check_data_quality(train_examples)
```

**4. Verify loss function:**

```python
# Wrong loss function for task
# Retrieval: use MultipleNegativesRankingLoss
# Classification: use ContrastiveLoss

# Test loss function manually
sample_batch = next(iter(train_dataloader))

# Calculate loss
loss_value = train_loss(sample_batch, model)
print(f"Sample batch loss: {loss_value}")

# Loss should be reasonable (not NaN, not infinity)
if np.isnan(loss_value) or np.isinf(loss_value):
    print("ERROR: Invalid loss value")
```

**5. Check gradient flow:**

```python
# Are gradients flowing through model?
def check_gradients(model):
    """Check if gradients are computed"""

    for name, param in model.named_parameters():
        if param.requires_grad:
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                print(f"{name}: grad_norm = {grad_norm:.6f}")
            else:
                print(f"{name}: NO GRADIENT")

# After one training step
train_step(sample_batch)
check_gradients(model)

# All parameters should have gradients
```

**6. Try simpler baseline:**

```python
# Can model overfit on small dataset?
# If not, something is fundamentally wrong

# Train on tiny dataset
tiny_dataset = train_examples[:100]
tiny_dataloader = DataLoader(tiny_dataset, shuffle=True, batch_size=16)

model.fit(
    train_objectives=[(tiny_dataloader, train_loss)],
    epochs=10  # Many epochs
)

# Evaluate on same tiny dataset
score = evaluate(model, tiny_dataset)
print(f"Score on tiny dataset: {score}")

# Should achieve very high score (overfitting)
# If not, model or loss function has issues
```

---

## Trade-offs & Decisions (Q26-Q30)

### Q26: Fine-tune one general model vs multiple specialized models?

**Answer:**

**Comparison:**

| Approach | Pros | Cons | When to use |
|----------|------|------|-------------|
| **One general model** | - Simple deployment<br>- One model to maintain<br>- Works across domains | - May not excel in any domain<br>- Larger training data needed | - Similar domains<br>- Limited resources<br>- Simple use case |
| **Multiple specialized** | - Best performance per domain<br>- Smaller training data per model<br>- Easier to improve specific area | - Complex deployment<br>- Higher maintenance cost<br>- Need routing logic | - Very different domains<br>- Performance critical<br>- Abundant data |

**Example scenario:**

```python
# Company has support tickets for:
# - Technical issues
# - Billing questions
# - Account management

# Approach 1: One general model
general_model = SentenceTransformer('all-MiniLM-L6-v2')

# Train on all data mixed
all_data = technical_data + billing_data + account_data
train_model(general_model, all_data)

# Simple inference
query_emb = general_model.encode(query)
results = search(query_emb, all_corpus)

# Performance: 82% accuracy across all domains

# Approach 2: Three specialized models
tech_model = train_model(base_model, technical_data)      # 91% on technical
billing_model = train_model(base_model, billing_data)     # 93% on billing
account_model = train_model(base_model, account_data)     # 89% on account

# Need classifier for routing
def route_query(query):
    category = classifier.predict(query)

    if category == 'technical':
        return tech_model
    elif category == 'billing':
        return billing_model
    else:
        return account_model

# More complex but better performance
model = route_query(query)
query_emb = model.encode(query)
results = search(query_emb, domain_specific_corpus)
```

**Decision framework:**

```python
def choose_approach(domains, data_per_domain, performance_requirement):
    """
    Decide between general vs specialized models

    Args:
        domains: Number of different domains
        data_per_domain: Training examples per domain
        performance_requirement: Required accuracy

    Returns:
        'general' or 'specialized'
    """

    # If very few domains and similar, go general
    if domains <= 2:
        return 'general'

    # If limited data per domain, go general
    if data_per_domain < 2000:
        return 'general'

    # If performance is critical and have data, go specialized
    if performance_requirement > 0.85 and data_per_domain >= 5000:
        return 'specialized'

    # Default: general (simpler)
    return 'general'

# Example
decision = choose_approach(
    domains=3,
    data_per_domain=8000,
    performance_requirement=0.90
)
# Returns: 'specialized'
```

**Hybrid approach:**

```python
# Best of both worlds: general model with domain adaptation

# 1. Train general model on all data
general_model = train_model(base_model, all_data)

# 2. Fine-tune copies for each domain (few epochs)
tech_model = clone_and_finetune(general_model, technical_data, epochs=1)
billing_model = clone_and_finetune(general_model, billing_data, epochs=1)

# 3. Use general as fallback
def hybrid_search(query):
    category = classifier.predict(query)
    confidence = classifier.predict_proba(query).max()

    # High confidence: use specialized
    if confidence > 0.8:
        model = get_specialized_model(category)
    else:
        # Low confidence: use general
        model = general_model

    return model.encode(query)

# Benefits:
# - Better performance on clear cases
# - Fallback for ambiguous cases
# - Moderate complexity
```

---

### Q27: When to fine-tune vs use cross-encoder for reranking?

**Answer:**

**Comparison:**

**Fine-tuned bi-encoder (sentence-transformers):**
- Fast at scale (pre-compute embeddings)
- Good for first-stage retrieval
- Less accurate than cross-encoder
- Suitable for: initial retrieval from large corpus

**Cross-encoder:**
- Slow (must encode each pair)
- Very accurate
- Not suitable for large-scale search
- Suitable for: reranking top results

**Best practice: Combine both**

```python
from sentence_transformers import SentenceTransformer, CrossEncoder

# Stage 1: Fast retrieval with fine-tuned bi-encoder
bi_encoder = SentenceTransformer('./fine_tuned_model')

query = "How to reset password?"
query_emb = bi_encoder.encode(query)

# Retrieve top 100 from 1M documents (fast)
candidates = faiss_search(query_emb, corpus_embeddings, top_k=100)

# Stage 2: Accurate reranking with cross-encoder
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

# Score each candidate (slow but accurate)
pairs = [[query, doc] for doc in candidates]
scores = cross_encoder.predict(pairs)

# Rerank
ranked_indices = np.argsort(scores)[::-1]
final_results = [candidates[i] for i in ranked_indices[:10]]
```

**When to use what:**

```python
# Use fine-tuned bi-encoder ONLY if:
if corpus_size > 100000 and latency_budget_ms < 100:
    approach = 'bi_encoder_only'
    # Fast but less accurate

# Use cross-encoder ONLY if:
elif corpus_size < 1000 and latency_budget_ms > 500:
    approach = 'cross_encoder_only'
    # Accurate but slow

# Use both (recommended) if:
elif corpus_size > 10000 and latency_budget_ms < 500:
    approach = 'bi_encoder_then_cross_encoder'
    # Best of both worlds

# Performance comparison
comparison = {
    'bi_encoder_only': {
        'speed': '50ms for 1M docs',
        'accuracy': '0.82 nDCG@10',
        'cost': 'Low'
    },
    'cross_encoder_only': {
        'speed': '5000ms for 1K docs',
        'accuracy': '0.94 nDCG@10',
        'cost': 'High'
    },
    'hybrid': {
        'speed': '150ms for 1M docs',  # 50ms retrieval + 100ms rerank
        'accuracy': '0.92 nDCG@10',
        'cost': 'Medium'
    }
}
```

**Implementation:**

```python
class HybridRetrieval:
    """Combine bi-encoder retrieval with cross-encoder reranking"""

    def __init__(self, bi_encoder_path, cross_encoder_path):
        self.bi_encoder = SentenceTransformer(bi_encoder_path)
        self.cross_encoder = CrossEncoder(cross_encoder_path)

    def search(self, query, corpus, corpus_embeddings, top_k=10, rerank_top_n=100):
        """
        Two-stage retrieval

        Args:
            query: Search query
            corpus: List of documents
            corpus_embeddings: Pre-computed embeddings
            top_k: Final number of results
            rerank_top_n: How many to rerank

        Returns:
            Top-k documents
        """
        # Stage 1: Fast retrieval
        query_emb = self.bi_encoder.encode(query)
        similarities = cosine_similarity([query_emb], corpus_embeddings)[0]
        top_indices = np.argsort(similarities)[-rerank_top_n:][::-1]

        candidates = [corpus[i] for i in top_indices]

        # Stage 2: Accurate reranking
        pairs = [[query, doc] for doc in candidates]
        scores = self.cross_encoder.predict(pairs)

        # Final ranking
        reranked_indices = np.argsort(scores)[::-1][:top_k]
        results = [candidates[i] for i in reranked_indices]

        return results

# Usage
retrieval = HybridRetrieval(
    bi_encoder_path='./fine_tuned_bi_encoder',
    cross_encoder_path='cross-encoder/ms-marco-MiniLM-L-12-v2'
)

results = retrieval.search(query, corpus, corpus_embeddings)
```

---

### Q28: Fine-tune from scratch vs continue from pre-trained?

**Answer:**

**Almost always continue from pre-trained.**

**Comparison:**

| Approach | Training time | Data needed | Performance | When to use |
|----------|--------------|-------------|-------------|-------------|
| **From scratch** | Weeks | Millions | Poor | Never (for most cases) |
| **From pre-trained** | Hours | Thousands | Good | Always (default) |

**Why pre-trained is better:**

```python
# Training from scratch
model_scratch = train_from_scratch(
    training_data=1_000_000,  # Need millions
    training_time='2 weeks',
    gpu_cost=5000,
    final_performance=0.75    # Likely worse than pre-trained
)

# Fine-tuning from pre-trained
model_finetuned = finetune_pretrained(
    base_model='all-MiniLM-L6-v2',  # Already trained on billions
    training_data=10_000,            # Need thousands
    training_time='2 hours',
    gpu_cost=10,
    final_performance=0.87           # Better!
)
```

**Rare cases to train from scratch:**

```python
# 1. Completely new domain with unique tokenization
if language == 'ancient_sumerian':
    # Pre-trained models don't know this language
    train_from_scratch = True

# 2. Extreme privacy requirements
if cannot_use_public_models:
    # Can't use models trained on public data
    train_from_scratch = True

# 3. Very specific architecture needs
if need_custom_architecture:
    train_from_scratch = True

# Otherwise: ALWAYS use pre-trained
```

**Best practice:**

```python
from sentence_transformers import SentenceTransformer

# Start with best pre-trained model for your task
base_model = SentenceTransformer('all-MiniLM-L6-v2')

# Fine-tune on your data
# Model retains general knowledge + learns your domain
train_model(base_model, your_domain_data)

# Benefits:
# - Faster training
# - Less data needed
# - Better performance
# - Generalization to unseen examples
```

---

### Q29: Fine-tune entire model vs only train top layers?

**Answer:**

**For sentence-transformers: fine-tune entire model (default).**

**Comparison:**

| Approach | Speed | Performance | Memory | When to use |
|----------|-------|-------------|--------|-------------|
| **Freeze lower layers** | Faster | Worse | Less | Limited resources |
| **Fine-tune all layers** | Slower | Better | More | Default (recommended) |

**Implementation:**

```python
from sentence_transformers import SentenceTransformer

# Default: fine-tune all layers (recommended)
model = SentenceTransformer('all-MiniLM-L6-v2')

# All parameters will be updated during training
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=3
)

# Freeze lower layers (not recommended, but possible)
def freeze_layers(model, num_layers_to_freeze=6):
    """Freeze first N transformer layers"""

    # Access transformer model
    transformer = model[0].auto_model

    # Freeze embeddings
    for param in transformer.embeddings.parameters():
        param.requires_grad = False

    # Freeze first N layers
    for layer in transformer.encoder.layer[:num_layers_to_freeze]:
        for param in layer.parameters():
            param.requires_grad = False

    # Keep top layers trainable
    # (last few layers + pooling)

    return model

# Use frozen model
model = freeze_layers(model, num_layers_to_freeze=6)

# Training will be faster but likely less accurate
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=3
)
```

**When freezing makes sense:**

```python
# 1. Very limited training data
if training_examples < 500:
    # Freeze to prevent overfitting
    freeze_lower_layers = True

# 2. Domain is very similar to pre-training
if domain_similarity > 0.9:
    # Lower layers already good, only adapt top
    freeze_lower_layers = True

# 3. Extremely limited compute
if gpu_memory < 8:  # GB
    freeze_lower_layers = True

# Otherwise: train all layers
```

**Empirical results:**

```python
# Experiment: Support ticket classification

# Fine-tune all layers
all_layers_model = finetune_all(train_data)
# Accuracy: 0.87
# Training time: 45 min

# Freeze 6 layers (of 12)
frozen_model = finetune_top_only(train_data, freeze=6)
# Accuracy: 0.82  # 5% worse
# Training time: 25 min  # Faster

# Freeze 9 layers
frozen_model = finetune_top_only(train_data, freeze=9)
# Accuracy: 0.78  # 9% worse
# Training time: 15 min

# Verdict: Train all layers unless you have constraints
```

**Learning rate strategy for all layers:**

```python
# Use discriminative learning rates (optional)
# Lower layers: smaller LR (already good)
# Upper layers: larger LR (need more adaptation)

# sentence-transformers doesn't directly support this
# But default settings work well for most cases
```

---

### Q30: When to retrain vs incrementally update the model?

**Answer:**

**Decision framework:**

| Scenario | Approach | Reason |
|----------|----------|--------|
| **New domain data** | Retrain from scratch | Domain shift, old data may hurt |
| **More data from same domain** | Incremental update | Preserve existing knowledge |
| **Significant data distribution change** | Retrain | Old model may be outdated |
| **Minor updates/corrections** | Incremental | Efficient, fast deployment |
| **Major schema change** | Retrain | Fundamental structure changed |

**Retrain from scratch:**

```python
# Scenarios:
# 1. Domain shift
old_domain = 'technical_support'
new_domain = 'medical_diagnosis'  # Completely different
# Decision: Retrain from base model

# 2. Old data is outdated
if data_age > 2_years:
    # Technology/language has changed
    # Decision: Retrain with recent data only

# 3. Performance degradation
if current_performance < 0.7:
    # Model is no longer useful
    # Decision: Start fresh

# Implementation
model = SentenceTransformer('all-MiniLM-L6-v2')  # Fresh start
train_model(model, new_data_only)
```

**Incremental update:**

```python
# Scenarios:
# 1. More examples from same domain
existing_data = load_old_training_data()  # 10K examples
new_data = load_new_training_data()        # 2K new examples

# Decision: Incremental update

# 2. Fix specific errors
problem_queries = identify_failures(model)
corrected_data = create_training_from_failures(problem_queries)

# Decision: Incremental update

# Implementation
model = SentenceTransformer('./current_production_model')

# Mix old and new (experience replay)
mixed_data = new_data + sample(existing_data, len(new_data))

# Train with lower LR and fewer epochs
model.fit(
    train_objectives=[(mixed_dataloader, train_loss)],
    epochs=1,  # Few epochs
    optimizer_params={'lr': 1e-5}  # Lower LR
)
```

**Hybrid: Periodic retraining + incremental updates:**

```python
class ModelUpdateStrategy:
    """Manage model updates over time"""

    def __init__(self, retrain_interval_days=90):
        self.retrain_interval = retrain_interval_days
        self.last_retrain = datetime.now()
        self.incremental_data = []

    def should_retrain(self):
        """Decide if it's time for full retrain"""
        days_since_retrain = (datetime.now() - self.last_retrain).days

        # Retrain every 3 months
        if days_since_retrain > self.retrain_interval:
            return True

        # Or if accumulated too much incremental data
        if len(self.incremental_data) > 50000:
            return True

        return False

    def update_model(self, new_data):
        """Decide how to update model"""

        if self.should_retrain():
            print("Performing full retrain")

            # Collect all data
            all_data = load_historical_data() + self.incremental_data + new_data

            # Retrain from base
            model = SentenceTransformer('all-MiniLM-L6-v2')
            train_model(model, all_data, epochs=3)

            # Reset
            self.last_retrain = datetime.now()
            self.incremental_data = []

        else:
            print("Performing incremental update")

            # Load current model
            model = SentenceTransformer('./current_model')

            # Incremental training
            incremental_train(model, new_data, epochs=1)

            # Track accumulated data
            self.incremental_data.extend(new_data)

        return model

# Usage
strategy = ModelUpdateStrategy(retrain_interval_days=90)

# Week 1: new data arrives
model = strategy.update_model(week1_data)  # Incremental

# Week 2: more data
model = strategy.update_model(week2_data)  # Incremental

# ...

# Week 13: 3 months passed
model = strategy.update_model(week13_data)  # Full retrain
```

**Best practices:**

```python
# 1. Version all models
save_model(model, version='v1.0.0')

# 2. A/B test before deployment
ab_test(old_model='v1.0.0', new_model='v1.1.0', traffic_split=0.1)

# 3. Keep training data history
archive_training_data(new_data, timestamp=now())

# 4. Monitor performance over time
track_metrics(model, version, timestamp, metrics)

# 5. Rollback capability
if new_model_performance < old_model_performance:
    rollback_to_version('v1.0.0')
```

---

## Additional Resources

- [Sentence-Transformers Documentation](https://www.sbert.net/)
- [Training Tutorial](https://www.sbert.net/docs/training/overview.html)
- [Loss Functions Guide](https://www.sbert.net/docs/package_reference/losses.html)
- [Fine-tuning Examples](https://github.com/UKPLab/sentence-transformers/tree/master/examples/training)
