"""
Address Parsing and Normalization Module.

Responsible for:
- Standardizing street abbreviations and descriptors (St, Ave, Rd, Blvd, etc.)
- Parsing building numbers, postal codes, and secondary unit designations (Ste, Apt, Fl)
- Cleaning and canonicalizing address strings for cross-record matching
"""

import re
from typing import Dict, Optional, Tuple


# Standard street type / unit abbreviation mappings
STREET_ABBREVIATIONS: Dict[str, str] = {
    "street": "st",
    "avenue": "ave",
    "road": "rd",
    "boulevard": "blvd",
    "drive": "dr",
    "lane": "ln",
    "way": "way",
    "court": "ct",
    "place": "pl",
    "circle": "cir",
    "highway": "hwy",
    "parkway": "pkwy",
    "square": "sq",
    "terrace": "ter",
    "suite": "ste",
    "apartment": "apt",
    "building": "bldg",
    "floor": "fl",
    "department": "dept",
    "post office box": "po box",
    "p.o. box": "po box",
    "p o box": "po box",
    "pobox": "po box",
}

# Regex to standardize street abbreviations at word boundaries
_STREET_REPLACEMENTS = [
    (re.compile(rf"\b{re.escape(k)}\b", re.IGNORECASE), v)
    for k, v in STREET_ABBREVIATIONS.items()
]

# Regex for extracting postal code patterns (5-digit US, UK alphanumeric, 5-digit EU)
_POSTAL_REGEX = re.compile(
    r"\b([A-Z]{1,2}\d[A-Z\d]?\s*\d[A-Z]{2}|\d{5}(?:-\d{4})?|\d{4,6})\b",
    re.IGNORECASE,
)

# Regex for extracting leading street numbers
_STREET_NUMBER_REGEX = re.compile(r"^\s*(\d+[\w\-]*)")


def clean_address(address: Optional[str]) -> str:
    """
    Standardize an address string:
    - Lowercase and normalize whitespace
    - Standardize common street and unit abbreviations
    - Strip trailing/leading punctuation
    """
    if not address or not isinstance(address, str):
        return ""

    addr = address.lower().strip()
    # Remove special punctuation except hyphens and slashes
    addr = re.sub(r"[^\w\s\-/]", " ", addr)

    # Standardize abbreviations
    for pattern, replacement in _STREET_REPLACEMENTS:
        addr = pattern.sub(replacement, addr)

    # Collapse whitespace
    addr = re.sub(r"\s+", " ", addr).strip()
    return addr


def extract_postal_code(text: Optional[str]) -> Optional[str]:
    """Extract standard postal/ZIP code if detected in text."""
    if not text or not isinstance(text, str):
        return None
    match = _POSTAL_REGEX.search(text)
    if match:
        return re.sub(r"\s+", "", match.group(1)).upper()
    return None


def extract_street_number(address: Optional[str]) -> Optional[str]:
    """Extract leading street or building number from an address string."""
    if not address or not isinstance(address, str):
        return None
    match = _STREET_NUMBER_REGEX.search(address)
    if match:
        return match.group(1).strip()
    return None


def parse_address_components(address: Optional[str]) -> Dict[str, Optional[str]]:
    """
    Parse an address into core components:
    - clean_address
    - street_number
    - postal_code
    """
    cleaned = clean_address(address)
    return {
        "address_cleaned": cleaned,
        "street_number": extract_street_number(cleaned),
        "postal_code": extract_postal_code(address or ""),
    }
