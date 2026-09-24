"""
Legal Suffixes and Corporate Form Normalization.

Defines multi-jurisdictional legal and business suffix mappings and provides
logic for parsing, stripping, and standardizing corporate suffixes.
Supported jurisdictions include US, UK, Germany, France, Spain, Italy,
Netherlands, and Commonwealth nations.
"""

import re
from typing import Dict, List, Optional, Set, Tuple


# Canonical mapping of corporate designations and variations to standard forms
LEGAL_SUFFIX_MAPPINGS: Dict[str, str] = {
    # Limited / Ltd
    "limited": "ltd",
    "ltd": "ltd",
    "pty ltd": "ltd",
    "proprietary limited": "ltd",
    "unlimited": "unltd",
    "public limited company": "plc",
    "plc": "plc",

    # Incorporation / Company
    "incorporated": "inc",
    "inc": "inc",
    "corporation": "corp",
    "corp": "corp",
    "company": "co",
    "co": "co",

    # Limited Liability
    "limited liability company": "llc",
    "llc": "llc",
    "l l c": "llc",
    "limited liability partnership": "llp",
    "llp": "llp",
    "l l p": "llp",
    "lp": "lp",
    "limited partnership": "lp",

    # Germany / Austria / Switzerland (DACH)
    "gmbh": "gmbh",
    "g m b h": "gmbh",
    "gmbh and co kg": "gmbh_co_kg",
    "gmbh and co": "gmbh_co",
    "aktiengesellschaft": "ag",
    "ag": "ag",
    "kommanditgesellschaft": "kg",
    "kg": "kg",
    "unternehmergesellschaft": "ug",
    "ug": "ug",
    "eingetragener kaufmann": "ek",
    "ek": "ek",
    "eingetragener verein": "ev",
    "ev": "ev",

    # France / Belgium / Romance
    "societe anonyme": "sa",
    "sa": "sa",
    "s a": "sa",
    "societe par actions simplifiee": "sas",
    "sas": "sas",
    "societe a responsabilite limitee": "sarl",
    "sarl": "sarl",
    "societe en nom collectif": "snc",
    "snc": "snc",

    # Spain / Latin America
    "sociedad anonima": "sa",
    "sociedad limitada": "sl",
    "sl": "sl",
    "s l": "sl",
    "sociedad de responsabilidad limitada": "srl",
    "srl": "srl",
    "sa de cv": "sa_de_cv",
    "s a de c v": "sa_de_cv",

    # Italy
    "societa per azioni": "spa",
    "spa": "spa",
    "s p a": "spa",

    # Netherlands
    "besloten vennootschap": "bv",
    "bv": "bv",
    "b v": "bv",
    "naamloze vennootschap": "nv",
    "nv": "nv",
    "n v": "nv",

    # Nordic
    "aktiebolag": "ab",
    "ab": "ab",
    "aksjeselskap": "as",
    "as": "as",
    "osakeyhtio": "oy",
    "oy": "oy",
}

# Sort patterns by decreasing length so multi-word suffixes (e.g. 'public limited company')
# match before single-word suffixes (e.g. 'company' or 'ltd')
SORTED_SUFFIXES: List[str] = sorted(
    LEGAL_SUFFIX_MAPPINGS.keys(), key=lambda s: len(s), reverse=True
)

# Precompile regex pattern anchored at word boundary and end of string
# Allows trailing whitespace/punctuation
_SUFFIX_PATTERN_STR = (
    r"(?:\b|_)(?:" + "|".join(re.escape(s) for s in SORTED_SUFFIXES) + r")\s*$"
)
_SUFFIX_REGEX = re.compile(_SUFFIX_PATTERN_STR, flags=re.IGNORECASE)


def extract_legal_suffix(name: str) -> Optional[str]:
    """
    Extract the legal suffix from an entity name if present.
    Returns normalized canonical suffix (e.g. 'limited' -> 'ltd') or None.
    """
    if not isinstance(name, str) or not name.strip():
        return None

    cleaned = name.lower().strip()
    # Strip trailing punctuation before checking
    cleaned = re.sub(r"[\.,;:!\-]+$", "", cleaned).strip()

    match = _SUFFIX_REGEX.search(cleaned)
    if match:
        raw_suffix = match.group(0).strip().strip("._- ")
        return LEGAL_SUFFIX_MAPPINGS.get(raw_suffix, raw_suffix)
    return None


def strip_legal_suffixes(name: str) -> str:
    """
    Remove any trailing legal/business suffixes from an entity name.
    Repeats recursively to handle compound suffixes (e.g., 'Corp LLC').
    """
    if not isinstance(name, str) or not name.strip():
        return ""

    current = name.strip()
    for _ in range(3):  # Max 3 passes to handle stacked suffixes
        # Normalize trailing dots/commas
        normalized_trailing = re.sub(r"[\.,;:!\-]+$", "", current).strip()
        match = _SUFFIX_REGEX.search(normalized_trailing)
        if match:
            current = normalized_trailing[: match.start()].strip()
            # Remove any trailing commas or conjunctions left behind
            current = re.sub(r"(?:,\s*|\s+and\s*|\s*&\s*)$", "", current).strip()
        else:
            break

    return current


def are_suffixes_compatible(suffix_a: Optional[str], suffix_b: Optional[str]) -> bool:
    """
    Check if two legal suffixes are compatible.
    If either is missing, returns True (absence of suffix doesn't strictly contradict).
    If both are present, checks if they match canonically.
    """
    if not suffix_a or not suffix_b:
        return True
    return suffix_a.lower() == suffix_b.lower()
