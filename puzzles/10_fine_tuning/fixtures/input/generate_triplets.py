#!/usr/bin/env python3
"""Generate extended triplet training dataset"""
import json
import random

# Define query categories with multiple examples
categories = {
    "password_reset": {
        "anchors": [
            "How do I reset my password?",
            "Can't remember my password",
            "Forgot password steps",
            "Password recovery help",
            "Reset account password",
            "Change forgotten password",
            "Recover lost password",
            "Password reset not working",
        ],
        "positives": [
            "Click 'Forgot Password' on login page and follow email instructions",
            "Use password reset link sent to your registered email",
            "Reset password from account settings security section",
            "Click forgot password and enter your email address",
            "Password reset instructions sent via email within minutes",
        ],
    },

    "ml_basics": {
        "anchors": [
            "What is machine learning?",
            "Explain ML to beginners",
            "Machine learning definition",
            "How does ML work?",
            "ML vs traditional programming",
            "Introduction to machine learning",
        ],
        "positives": [
            "Machine learning trains algorithms on data to make predictions",
            "ML enables computers to learn patterns without explicit programming",
            "Algorithms that improve automatically through experience with data",
            "Statistical techniques allowing computers to learn from examples",
        ],
    },

    "deep_learning": {
        "anchors": [
            "What is deep learning?",
            "Deep learning explained",
            "Neural network basics",
            "How do neural networks work?",
            "Deep vs shallow learning",
            "Introduction to deep learning",
        ],
        "positives": [
            "Deep learning uses multi-layer neural networks to learn hierarchies",
            "Neural networks with multiple hidden layers for complex patterns",
            "Subset of ML using layered neural networks",
            "Learning representations through multiple processing layers",
        ],
    },

    "transformers": {
        "anchors": [
            "What are transformers in NLP?",
            "Transformer architecture explained",
            "How do transformers work?",
            "BERT and GPT architecture",
            "Attention mechanism in transformers",
        ],
        "positives": [
            "Transformers use self-attention to process sequences in parallel",
            "Neural architecture using attention mechanisms for NLP",
            "Model architecture that revolutionized NLP with attention",
            "Processes entire sequence simultaneously using attention",
        ],
    },

    "embeddings": {
        "anchors": [
            "What are embeddings?",
            "Word embeddings explained",
            "Vector representations of text",
            "How do word2vec work?",
            "Text to vector conversion",
        ],
        "positives": [
            "Dense vector representations capturing semantic meaning",
            "Numerical vectors encoding text or word meaning",
            "Low-dimensional vectors representing high-dimensional data",
            "Vector space mapping of words preserving semantic relationships",
        ],
    },

    "api_auth": {
        "anchors": [
            "API authentication method",
            "How to authenticate API requests?",
            "API security best practices",
            "Bearer token usage",
            "API key generation",
        ],
        "positives": [
            "Use Bearer token in Authorization header for API requests",
            "Include API key in request headers for authentication",
            "OAuth2 or JWT tokens for secure API authentication",
            "Add authentication token to HTTP headers",
        ],
    },

    "database": {
        "anchors": [
            "Database connection error",
            "Can't connect to database",
            "Database timeout issues",
            "SQL connection failed",
            "Database not responding",
        ],
        "positives": [
            "Check database credentials and server accessibility",
            "Verify connection string and database server status",
            "Ensure database service is running and ports are open",
            "Check firewall rules and database permissions",
        ],
    },

    "deployment": {
        "anchors": [
            "Deploy to production",
            "Production deployment steps",
            "How to release new version?",
            "CI/CD pipeline setup",
            "Deployment best practices",
        ],
        "positives": [
            "Run tests, build artifacts, and deploy via CI/CD pipeline",
            "Execute deployment pipeline with health checks",
            "Use automated deployment with rollback capability",
            "Deploy through staging before production release",
        ],
    },

    "fine_tuning": {
        "anchors": [
            "What is fine-tuning?",
            "How to fine-tune models?",
            "Transfer learning explained",
            "Adapt pretrained models",
            "Domain adaptation in ML",
        ],
        "positives": [
            "Fine-tuning adapts pretrained models to specific tasks",
            "Continue training pretrained model on domain-specific data",
            "Transfer learning by updating model weights on new data",
            "Specialize general model for specific use case",
        ],
    },

    "rag": {
        "anchors": [
            "What is RAG?",
            "Retrieval augmented generation",
            "How does RAG work?",
            "RAG architecture explained",
            "Combining retrieval with generation",
        ],
        "positives": [
            "RAG retrieves relevant documents to enhance LLM responses",
            "Combines information retrieval with text generation",
            "Augments language model with retrieved knowledge",
            "Retrieves context from database before generating answer",
        ],
    },
}

# Generic negatives (unrelated topics)
generic_negatives = [
    "Annual subscription includes priority support",
    "Office hours are Monday through Friday 9am to 6pm",
    "Employee referral bonus is $2000 per hire",
    "Q4 revenue was $2.5 million with 35% growth",
    "Summer internship program accepts 20 students",
    "Quarterly all-hands meeting is every third Friday",
    "Free trial includes all premium features for 14 days",
    "Healthcare coverage starts first day of employment",
    "Professional plan costs $99/month for 25 users",
    "Vacation policy provides 15 days annually",
    "Remote work available for approved positions",
    "Company founded in 2015 by three engineers",
    "Stock options vest over 4 years with 1-year cliff",
    "Ergonomic desk setup includes standing desk",
    "Coffee and snacks provided in break rooms",
    "Holiday party scheduled for December 20th",
    "Team building events quarterly at various venues",
    "Performance reviews conducted biannually",
    "Parking validation available at front desk",
    "Gym membership discount for full-time employees",
]

def generate_triplets():
    triplets = []
    random.seed(42)

    # Get category names
    cat_names = list(categories.keys())

    # For each category, create multiple triplets
    for cat_name, cat_data in categories.items():
        anchors = cat_data["anchors"]
        positives = cat_data["positives"]

        # Get negatives from other categories
        other_cats = [c for c in cat_names if c != cat_name]

        # Create triplets by pairing anchors with positives
        for anchor in anchors:
            for positive in positives:
                # Use generic negative
                negative = random.choice(generic_negatives)
                triplets.append({
                    "anchor": anchor,
                    "positive": positive,
                    "negative": negative
                })

                # Also use negative from another category
                other_cat = random.choice(other_cats)
                other_positive = random.choice(categories[other_cat]["positives"])
                triplets.append({
                    "anchor": anchor,
                    "positive": positive,
                    "negative": other_positive
                })

    # Shuffle and return
    random.shuffle(triplets)
    return triplets

if __name__ == "__main__":
    triplets = generate_triplets()
    print(f"Generated {len(triplets)} triplets")

    # Save to file
    with open('training_triplets.json', 'w') as f:
        json.dump(triplets, f, indent=2)

    print(f"✓ Saved to training_triplets.json")
    print(f"\nExample triplet:")
    example = triplets[0]
    print(f"  Anchor: {example['anchor']}")
    print(f"  Positive: {example['positive']}")
    print(f"  Negative: {example['negative']}")
