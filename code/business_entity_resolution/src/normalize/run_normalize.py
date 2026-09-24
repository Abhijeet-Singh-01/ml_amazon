"""
Normalization Pipeline Runner.

Applies text cleaning, corporate suffix parsing/stripping, and address
normalization across source and target entity DataFrames to generate
standardized schemas for downstream blocking and feature engineering.
"""

from typing import Optional
import pandas as pd

from .text_clean import clean_text
from .legal_suffixes import extract_legal_suffix, strip_legal_suffixes
from .address_parse import clean_address, parse_address_components


def normalize_record_name(raw_name: Optional[str]) -> dict:
    """Normalize entity name, extracting base name and legal suffix."""
    cleaned = clean_text(raw_name, lowercase=True, strip_punctuation=True)
    suffix = extract_legal_suffix(cleaned)
    base_name = strip_legal_suffixes(cleaned)
    return {
        "name_clean": cleaned,
        "name_base": base_name,
        "legal_suffix": suffix or "",
    }


def normalize_entity_dataframe(
    df: pd.DataFrame,
    id_col: str = "id",
    name_col: str = "name",
    address_col: Optional[str] = "address",
    country_col: Optional[str] = "country",
    postal_col: Optional[str] = "postal_code",
) -> pd.DataFrame:
    """
    Apply comprehensive normalization to an entity DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input entity table.
    id_col : str
        Name of primary key identifier column.
    name_col : str
        Name of business entity name column.
    address_col : Optional[str]
        Name of street/physical address column.
    country_col : Optional[str]
        Name of country code/name column.
    postal_col : Optional[str]
        Name of postal/ZIP code column if pre-existing.

    Returns
    -------
    pd.DataFrame
        Normalized DataFrame with canonical columns:
        - `id`
        - `name_raw`
        - `name_clean`
        - `name_base`
        - `legal_suffix`
        - `address_raw`
        - `address_clean`
        - `postal_code`
        - `country_code`
    """
    res = pd.DataFrame()
    res["id"] = df[id_col].astype(str)
    res["name_raw"] = df[name_col].fillna("").astype(str)

    # Name normalization
    name_norms = [normalize_record_name(n) for n in res["name_raw"]]
    res["name_clean"] = [n["name_clean"] for n in name_norms]
    res["name_base"] = [n["name_base"] for n in name_norms]
    res["legal_suffix"] = [n["legal_suffix"] for n in name_norms]

    # Address normalization
    if address_col and address_col in df.columns:
        res["address_raw"] = df[address_col].fillna("").astype(str)
        parsed_addresses = [parse_address_components(a) for a in res["address_raw"]]
        res["address_clean"] = [p["address_cleaned"] for p in parsed_addresses]
        # Use existing postal col if available, else parsed
        if postal_col and postal_col in df.columns:
            res["postal_code"] = (
                df[postal_col]
                .fillna("")
                .astype(str)
                .str.strip()
                .str.upper()
            )
        else:
            res["postal_code"] = [p["postal_code"] or "" for p in parsed_addresses]
    else:
        res["address_raw"] = ""
        res["address_clean"] = ""
        res["postal_code"] = (
            df[postal_col].fillna("").astype(str).str.strip().str.upper()
            if postal_col and postal_col in df.columns
            else ""
        )

    # Country normalization
    if country_col and country_col in df.columns:
        res["country_code"] = (
            df[country_col]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.upper()
        )
    else:
        res["country_code"] = ""

    return res


def run_normalization(
    source_df: pd.DataFrame,
    target_df: pd.DataFrame,
    id_col: str = "id",
    name_col: str = "name",
    address_col: Optional[str] = "address",
    country_col: Optional[str] = "country",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run full normalization pipeline over both source and target datasets.
    """
    norm_source = normalize_entity_dataframe(
        source_df,
        id_col=id_col,
        name_col=name_col,
        address_col=address_col,
        country_col=country_col,
    )
    norm_target = normalize_entity_dataframe(
        target_df,
        id_col=id_col,
        name_col=name_col,
        address_col=address_col,
        country_col=country_col,
    )
    return norm_source, norm_target
