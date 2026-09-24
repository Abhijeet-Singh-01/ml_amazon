"""
Text Cleaning and Normalization Utilities.

Provides reusable functions for:
- Unicode normalization (NFKD decomposition and ASCII transliteration)
- Lowercasing and whitespace collapsing
- Punctuation and special character standardization
- Alphanumeric token sanitization
"""

import re
import unicodedata
from typing import Optional


def normalize_unicode(text: str) -> str:
    """Normalize Unicode characters, strip diacritics/accents, and convert to ASCII."""
    if not isinstance(text, str):
        return ""
    normalized = unicodedata.normalize("NFKD", text)
    ascii_bytes = normalized.encode("ASCII", "ignore")
    return ascii_bytes.decode("utf-8")


def collapse_whitespace(text: str) -> str:
    """Replace multiple consecutive whitespaces with a single space and strip borders."""
    if not isinstance(text, str):
        return ""
    return re.sub(r"\s+", " ", text).strip()


def replace_symbols(text: str) -> str:
    """Standardize common symbol representations (e.g. & -> and)."""
    if not isinstance(text, str):
        return ""
    text = text.replace("&", " and ")
    text = text.replace("@", " at ")
    text = text.replace("+", " plus ")
    text = text.replace("/", " ")
    return text


def remove_punctuation(text: str, keep_hyphens: bool = False) -> str:
    """Remove punctuation characters while preserving alphanumeric tokens."""
    if not isinstance(text, str):
        return ""
    if keep_hyphens:
        pattern = r"[^\w\s\-]"
    else:
        pattern = r"[^\w\s]"
    return re.sub(pattern, " ", text)


def clean_text(
    text: Optional[str],
    lowercase: bool = True,
    strip_punctuation: bool = True,
    standardize_symbols: bool = True,
) -> str:
    """
    Apply full text cleaning pipeline:
    1. Unicode normalization
    2. Lowercasing
    3. Symbol replacement
    4. Punctuation stripping
    5. Whitespace collapsing
    """
    if text is None or not isinstance(text, str):
        return ""

    cleaned = normalize_unicode(text)

    if lowercase:
        cleaned = cleaned.lower()

    if standardize_symbols:
        cleaned = replace_symbols(cleaned)

    if strip_punctuation:
        cleaned = remove_punctuation(cleaned)

    cleaned = collapse_whitespace(cleaned)
    return cleaned
