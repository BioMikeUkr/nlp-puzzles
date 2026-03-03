"""
Sample text processing module for testing exercises.
Students will write tests for these functions.
"""

import re
from collections import Counter


def clean_text(text: str) -> str:
    """Remove extra whitespace and strip leading/trailing spaces."""
    if not isinstance(text, str):
        raise TypeError(f"Expected str, got {type(text).__name__}")
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str, lowercase: bool = True) -> list[str]:
    """Split text into word tokens."""
    if not isinstance(text, str):
        raise TypeError(f"Expected str, got {type(text).__name__}")
    cleaned = clean_text(text)
    if not cleaned:
        return []
    tokens = cleaned.split()
    if lowercase:
        tokens = [t.lower() for t in tokens]
    return tokens


def count_words(text: str) -> dict[str, int]:
    """Count word frequencies in text."""
    tokens = tokenize(text, lowercase=True)
    return dict(Counter(tokens))


def extract_emails(text: str) -> list[str]:
    """Extract email addresses from text."""
    if not isinstance(text, str):
        raise TypeError(f"Expected str, got {type(text).__name__}")
    pattern = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
    return re.findall(pattern, text)


def truncate(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """Truncate text to max_length, adding suffix if truncated."""
    if max_length < 0:
        raise ValueError("max_length must be non-negative")
    if len(text) <= max_length:
        return text
    if max_length < len(suffix):
        return text[:max_length]
    return text[: max_length - len(suffix)] + suffix


def normalize_whitespace(text: str) -> str:
    """Replace all whitespace chars (tabs, newlines) with single spaces."""
    return re.sub(r"\s+", " ", text).strip()


def is_palindrome(text: str) -> bool:
    """Check if text is a palindrome (ignoring case and non-alphanumeric chars)."""
    cleaned = re.sub(r"[^a-zA-Z0-9]", "", text).lower()
    return cleaned == cleaned[::-1]


def mask_pii(text: str) -> str:
    """Mask email addresses and phone numbers in text."""
    # Mask emails
    text = re.sub(
        r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
        "[EMAIL]",
        text,
    )
    # Mask phone numbers (simple US format)
    text = re.sub(
        r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",
        "[PHONE]",
        text,
    )
    return text
