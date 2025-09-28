from typing import Dict, Optional, Tuple

import pandas as pd
import streamlit as st
import unicodedata

from config.settings import MDM_FILE
from modules.loader import load_excel


def _normalize_header(name: str) -> str:
    normalized = unicodedata.normalize("NFKD", str(name))
    return "".join(ch for ch in normalized if not unicodedata.combining(ch)).lower()


@st.cache_data(show_spinner=False)
def load_mda() -> Tuple[pd.DataFrame, Dict[str, str]]:
    """Load the Attribute and Facet sheets from the MDM workbook."""
    mda_df = load_excel(MDM_FILE, sheet_name="Attribute")
    facet_labels_df = load_excel(MDM_FILE, sheet_name="Facet")

    if mda_df is None:
        mda_df = pd.DataFrame()
    if facet_labels_df is None:
        facet_labels_df = pd.DataFrame()

    facet_labels: Dict[str, str] = {}
    if not facet_labels_df.empty and {"Name", "Label"}.issubset(facet_labels_df.columns):
        facet_labels = dict(zip(facet_labels_df["Name"], facet_labels_df["Label"]))

    return mda_df, facet_labels


@st.cache_data(show_spinner=False, hash_funcs={pd.DataFrame: lambda df: df.to_json(date_unit="s", orient="split")})
def attribute_label_map(mda_df: Optional[pd.DataFrame]) -> Dict[str, str]:
    """Return MUDU -> label mapping from the MDA sheet when available."""
    if mda_df is None:
        mda_df, _ = load_mda()

    if mda_df is None or mda_df.empty or "MUDU" not in mda_df.columns:
        return {}

    normalized_cols = {_normalize_header(col): col for col in mda_df.columns}
    label_col = None
    for candidate in ("label", "display", "libelle"):
        col = normalized_cols.get(candidate)
        if col:
            label_col = col
            break

    if label_col is None:
        return {}

    mapping: Dict[str, str] = {}
    for _, row in mda_df.iterrows():
        mudu = row.get("MUDU")
        if pd.isna(mudu):
            continue
        label = row.get(label_col)
        if pd.notna(label) and str(label).strip() != "":
            mapping[str(mudu)] = str(label)
    return mapping


@st.cache_data(show_spinner=False, hash_funcs={pd.DataFrame: lambda df: df.to_json(date_unit="s", orient="split")})
def infer_attr_type(attr: str, mda_df: Optional[pd.DataFrame]) -> str:
    """Infer attribute type from the MDA (int, float or str)."""
    try:
        if mda_df is None:
            mda_df, _ = load_mda()

        if mda_df is None or mda_df.empty or "MUDU" not in mda_df.columns:
            return "str"

        row = mda_df.loc[mda_df["MUDU"] == attr]
        if row.empty:
            return "str"

        normalized_cols = {_normalize_header(col): col for col in mda_df.columns}
        type_col = None
        for candidate in ("type", "attrtype", "datatype", "attributetype", "columntype"):
            col = normalized_cols.get(candidate)
            if col:
                type_col = col
                break

        if type_col is None:
            return "str"

        value = str(row.iloc[0][type_col]).strip().lower()
        if any(token in value for token in ("int", "integer")):
            return "int"
        if any(token in value for token in ("float", "double", "decimal", "number", "numeric", "real")):
            return "float"
    except Exception:
        pass
    return "str"


@st.cache_data(show_spinner=False, hash_funcs={pd.DataFrame: lambda df: df.to_json(date_unit="s", orient="split")})
def attribute_order(mda_df: Optional[pd.DataFrame]) -> Tuple[str, ...]:
    """Return the ordered list of columns based on the MDA order."""
    if mda_df is None:
        mda_df, _ = load_mda()

    if mda_df is None or mda_df.empty:
        return tuple()
    if not {"MUDU", "Order"}.issubset(mda_df.columns):
        return tuple()

    subset = mda_df[["MUDU", "Order"]].dropna(subset=["MUDU"]).copy()
    if subset.empty:
        return tuple()

    subset["__OrderNum__"] = pd.to_numeric(subset["Order"], errors="coerce")
    subset.sort_values(["__OrderNum__", "MUDU"], inplace=True, kind="mergesort")
    ordered_attrs = subset["MUDU"].tolist()
    return tuple(ordered_attrs)


def apply_attribute_order(df: pd.DataFrame, mda_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Reorder columns so that they follow the MDA order when possible."""
    if df is None or df.empty:
        return df

    ordered_attrs = attribute_order(mda_df)
    if not ordered_attrs:
        return df

    all_cols = list(df.columns)
    ordered: list[str] = []
    for attr in ordered_attrs:
        if attr in df.columns and attr not in ordered:
            ordered.append(attr)

    remaining = [col for col in all_cols if col not in ordered]
    ordered.extend(remaining)

    if "RepeFonct" in ordered:
        ordered.remove("RepeFonct")
        ordered.insert(0, "RepeFonct")

    return df[ordered]
