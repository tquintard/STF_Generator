import os
import shutil
from typing import Dict, Optional, Union

import pandas as pd
import streamlit as st
from openpyxl import load_workbook

from config.settings import OUTPUT_DIR, log, BASE_DIR


def _ensure_output_dir():
    try:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
    except Exception as e:
        log(f"Failed to ensure output dir {OUTPUT_DIR}: {e}", level="ERROR")


def _infer_attr_type(attr: str, mda_df: Optional[pd.DataFrame]) -> str:
    if mda_df is None or mda_df.empty or "MUDU" not in mda_df.columns:
        return "str"
    row = mda_df.loc[mda_df["MUDU"] == attr]
    if row.empty:
        return "str"
    type_col = None
    for c in ("Type", "AttrType", "DataType", "AttributeType", "ColumnType"):
        if c in mda_df.columns:
            type_col = c
            break
    if type_col is None:
        return "str"
    t = str(row.iloc[0][type_col]).strip().lower()
    if any(x in t for x in ("int", "integer")):
        return "int"
    if any(x in t for x in ("float", "double", "decimal", "number", "numeric", "real")):
        return "float"
    return "str"


def _to_number(val):
    if val is None:
        return None
    s = str(val).strip()
    if s == "" or s.lower() in {"nan", "none", "null", "-"}:
        return None
    s = s.replace("\u00A0", " ").replace(" ", "").replace("'", "")
    if "," in s and "." in s:
        s = s.replace(".", "").replace(",", ".")
    elif "," in s and "." not in s:
        s = s.replace(",", ".")
    try:
        return float(s)
    except Exception:
        return None


def _excel_safe(name: str) -> str:
    """Best-effort transform of a MUDU to an Excel Named Range compatible name."""
    import re
    s = str(name).strip()
    # Replace spaces and dashes by underscore
    s = re.sub(r"[\s\-]+", "_", s)
    # Remove invalid characters
    s = re.sub(r"[^A-Za-z0-9_]", "", s)
    # Must not start with a number
    if s and s[0].isdigit():
        s = f"_{s}"
    return s


def _iter_defined_names(wb):
    """Yield DefinedName objects across openpyxl versions (list/dict backends)."""
    dn = wb.defined_names
    # Newer/older API compatibility
    try:
        # openpyxl <= some versions
        for obj in dn.definedName:
            yield obj
        return
    except AttributeError:
        pass
    try:
        # dict-like container
        for obj in dn.values():
            # Some versions store lists per key (duplicates): flatten
            if isinstance(obj, (list, tuple)):
                for o in obj:
                    yield o
            else:
                yield obj
        return
    except Exception:
        pass
    try:
        for key in dn.keys():
            obj = dn.get(key)
            if isinstance(obj, (list, tuple)):
                for o in obj:
                    yield o
            elif obj is not None:
                yield obj
    except Exception:
        return


def _write_by_named_ranges(wb, values: Dict[str, str], mda_df: Optional[pd.DataFrame]):
    """Write values using workbook Named Ranges.

    Matching rules:
    - Try exact name match first.
    - Fallback to case-insensitive match.
    - If a name refers to a range, write to its top-left cell.

    Returns list of keys that were not matched to any named range.
    """
    # Build a case-insensitive map of defined names -> (sheet, cell)
    names_map: Dict[str, tuple] = {}
    for dn in _iter_defined_names(wb):
        if getattr(dn, 'type', None) != 'RANGE':
            continue
        dests = list(getattr(dn, 'destinations', []) or [])
        if not dests:
            continue
        sheet_name, ref = dests[0]
        # If ref is a range like A1:B2, take top-left cell
        cell = ref.split(':', 1)[0]
        key = dn.name.strip()
        names_map[key.lower()] = (sheet_name, cell)
        safe = _excel_safe(key)
        if safe and safe not in names_map:
            names_map[safe] = (sheet_name, cell)
            names_map[safe.lower()] = (sheet_name, cell)

    for key, v in values.items():
        target = (
            names_map.get(key)
            or names_map.get(str(key).lower())
            or names_map.get(_excel_safe(key))
            or names_map.get(_excel_safe(key).lower())
        )
        if not target:
            continue
        sheet_name, cell = target
        if sheet_name not in wb.sheetnames:
            continue
        ws = wb[sheet_name]
        cell_obj = ws[cell]
        t = _infer_attr_type(key, mda_df)
        if t in ("int", "float"):
            num = _to_number(v)
            cell_obj.value = num if num is not None else v
        else:
            cell_obj.value = v


def _get_defined_base_names(wb) -> set:
    """Return the set of base defined names (excluding Excel built-ins)."""
    names = set()
    for dn in _iter_defined_names(wb):
        name = getattr(dn, 'name', '') or ''
        nlow = name.lower()
        if nlow.startswith('_xlnm.') or nlow in ('print_area', 'print_titles'):
            continue
        names.add(name)
    return names


def export_stf_from_row(selected: Union[pd.DataFrame, Dict]) -> str:
    """Duplicate the STF template and fill it with the selected ECS values.

    Returns the output filepath.
    """
    template_path = os.path.join(BASE_DIR, "template", "stf_templates.xlsx")
    if not os.path.exists(template_path):
        raise FileNotFoundError(f"Template not found: {template_path}")

    # Support both a single-row DataFrame or a dict (AgGrid selected row)
    if isinstance(selected, pd.DataFrame):
        row = selected.iloc[0]
        ecs = str(row.get("RepeFonct", "")).strip()
        values = {str(k): ("" if v is None else str(v))
                  for k, v in row.items()}
    elif isinstance(selected, dict):
        ecs = str(selected.get("RepeFonct", "")).strip()
        values = {str(k): ("" if v is None else str(v))
                  for k, v in selected.items()}
    else:
        raise TypeError(
            "selected must be a pandas DataFrame (1 row) or a dict of values")

    safe_ecs = "".join(c for c in ecs if c.isalnum()
                       or c in ("-", "_")) or "ECS"

    _ensure_output_dir()
    output_path = os.path.join(OUTPUT_DIR, f"{safe_ecs}.xlsx")
    shutil.copyfile(template_path, output_path)

    wb = load_workbook(output_path)

    mda_df = st.session_state.get("mda_df")

    # Enforce: only write attributes defined in MDA (template should be subset of MDA)
    if isinstance(mda_df, pd.DataFrame) and not mda_df.empty and 'MUDU' in mda_df.columns:
        mda_keys = set(mda_df['MUDU'].astype(str))
        values = {k: v for k, v in values.items() if k in mda_keys}

    # Write via named ranges matching MUDU names
    missing = _write_by_named_ranges(wb, values, mda_df)

    wb.save(output_path)
    log(f"STF exported for {ecs} -> {output_path}")

    # Report template names that are not in MDA
    try:
        if isinstance(mda_df, pd.DataFrame) and not mda_df.empty and 'MUDU' in mda_df.columns:
            defined_names = _get_defined_base_names(wb)
            mda_variants = {s.lower() for s in mda_df['MUDU'].astype(str)}
            mda_variants |= {_excel_safe(s).lower()
                             for s in mda_df['MUDU'].astype(str)}
            extras = [n for n in defined_names if n.lower()
                      not in mda_variants]
            if extras:
                st.info(
                    f"Noms définis dans le template non présents dans le MDA: {', '.join(sorted(extras))}")
    except Exception:
        pass
    return output_path
