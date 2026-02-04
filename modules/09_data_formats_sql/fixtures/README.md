# Module 9 Fixtures: Data Formats & SQL

## Overview

This directory contains sample datasets for practicing data format conversions and SQL operations.

## Input Data

### tickets.csv
Support ticket dataset with 50 records.

**Schema:**
- `ticket_id` (int): Unique ticket identifier
- `user_id` (int): User who created the ticket
- `category` (str): Technical, Billing, or Account
- `description` (str): Issue description
- `created_at` (datetime): Ticket creation timestamp
- `resolved_at` (datetime): Resolution timestamp
- `priority` (str): low, medium, high, critical
- `status` (str): All tickets are "resolved"

**Date range:** January 2024 - May 2024

**Size:** ~5KB

### users.csv
User information with 21 records.

**Schema:**
- `user_id` (int): Unique user identifier
- `username` (str): Username
- `email` (str): Email address
- `signup_date` (date): Account creation date
- `plan_type` (str): free, premium, enterprise
- `company` (str): Company name

**Size:** ~1.5KB

## Expected Output (created by notebooks)

### Parquet Files
- `tickets.parquet` - Compressed Parquet version (~2KB with Snappy)
- `tickets_partitioned/` - Directory with partitioned Parquet files by month and category

### Database Files
- `tickets.db` - SQLite database with tickets and users tables

## Usage in Tasks

### Task 1: Data Pipeline
- Load `tickets.csv`
- Convert to Parquet with various compressions
- Create partitioned Parquet dataset
- Load into SQLite database

### Task 2: SQL Analytics
- Load both CSV files into SQLite
- Perform SQL queries and analytics
- Practice JOINs, aggregations, window functions

## Generating Your Own Fixtures

You can create larger datasets using the code in the learning notebooks:

```python
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Generate synthetic tickets
np.random.seed(42)
n_tickets = 1000

tickets = pd.DataFrame({
    'ticket_id': range(1, n_tickets + 1),
    'user_id': np.random.randint(1, 100, n_tickets),
    'category': np.random.choice(['Technical', 'Billing', 'Account'], n_tickets),
    'description': [f'Issue description {i}' for i in range(n_tickets)],
    'created_at': [datetime(2024, 1, 1) + timedelta(days=np.random.randint(0, 150))
                   for _ in range(n_tickets)],
    'priority': np.random.choice(['low', 'medium', 'high', 'critical'], n_tickets),
    'status': 'resolved'
})

tickets['resolved_at'] = tickets['created_at'] + pd.to_timedelta(
    np.random.randint(1, 72, n_tickets), unit='h'
)

tickets.to_csv('fixtures/input/tickets_large.csv', index=False)
```
