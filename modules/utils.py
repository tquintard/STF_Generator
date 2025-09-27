# modules/utils.py

import io
import os
import re
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import streamlit as st
from config.settings import log
from st_aggrid import AgGrid, GridOptionsBuilder

from modules.exporter import export_stf_from_row
from modules.ui_modules import id_card_columns


def _attribute_label_map(mda_df: Optional[pd.DataFrame]) -> Dict[str, str]:
    """Return MUDU -> label mapping from the MDA sheet when available."""
    if mda_df is None or mda_df.empty or "MUDU" not in mda_df.columns:
        return {}
    label_col = None
    for candidate in ("Label", "Display", "Libelle", "Libelle"):
        if candidate in mda_df.columns:
            label_col = candidate
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


def _infer_attr_type(attr: str, mda_df: Optional[pd.DataFrame]) -> str:
    """Infer attribute type from the MDA (int, float or str)."""
    try:
        if mda_df is None or mda_df.empty or "MUDU" not in mda_df.columns:
            return "str"
        row = mda_df.loc[mda_df["MUDU"] == attr]
        if row.empty:
            return "str"
        type_col = None
        for candidate in ("Type", "AttrType", "DataType", "AttributeType", "ColumnType"):
            if candidate in mda_df.columns:
                type_col = candidate
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


def _to_number(val):
    """Parse a value to float when possible, handling FR/EN formats."""
    if val is None:
        return None
    s = str(val).strip()
    if not s or s.lower() in {"nan", "none", "null", "-"}:
        return None
    s = s.replace("\u00A0", " ").replace(" ", "").replace("'", "")
    if "," in s and "." in s:
        s = s.replace(".", "").replace(",", ".")
    elif "," in s:
        s = s.replace(",", ".")
    try:
        return float(s)
    except Exception:
        return None


def _format_number(num, kind: str) -> str:
    if num is None:
        return ""
    if kind == "int":
        try:
            return str(int(round(num)))
        except Exception:
            return str(num)
    formatted = f"{num:.6f}".rstrip("0").rstrip(".")
    return formatted


def _slugify(name: str) -> str:
    stem = Path(name).stem
    slug = re.sub(r"[^0-9A-Za-z]+", "_", stem).strip("_")
    return slug or "input"


def _search_patterns(query: str) -> list[str]:
    """Split search query on commas and convert * wildcards to regex."""
    if not query:
        return []
    patterns: list[str] = []
    for part in query.split(','):
        term = part.strip()
        if not term:
            continue
        escaped = re.escape(term)
        escaped = escaped.replace('\\*', '.*')
        patterns.append(escaped)
    return patterns


def apply_attribute_order(df: pd.DataFrame, mda_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Reorder columns so that they follow the MDA order when possible."""
    if df is None or df.empty:
        return df
    if mda_df is None or mda_df.empty:
        return df
    if not {"MUDU", "Order"}.issubset(mda_df.columns):
        return df
    all_cols = list(df.columns)
    subset = mda_df[mda_df["MUDU"].isin(all_cols)].copy()
    if subset.empty:
        return df
    subset["__OrderNum__"] = pd.to_numeric(subset["Order"], errors="coerce")
    subset.sort_values(["__OrderNum__", "MUDU"],
                       inplace=True, kind="mergesort")
    ordered_attrs = subset["MUDU"].tolist()
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


def smart_merge(df_left: pd.DataFrame, df_right: pd.DataFrame, on: str = "RepeFonct",
                label_left: str = "1D", label_right: str = "2D") -> pd.DataFrame:
    """Merge the left/right DataFrames and reconcile conflicting values."""
    merged = pd.merge(
        df_left,
        df_right,
        on=on,
        how="outer",
        suffixes=(f"_{label_left}", f"_{label_right}"),
        indicator=True,
    )
    existing_keys = set(merged[on].astype(str))

    def compute_ecs_mtr(value: str):
        s = str(value)
        if s.endswith("RA-"):
            candidate = s[:-1] + "M"
            if candidate in existing_keys:
                return candidate
        return None

    merged["ECS_Mtr"] = merged[on].apply(compute_ecs_mtr)
    shared_columns = set(df_left.columns).intersection(df_right.columns) - {on}
    mda_df = st.session_state.get("mda_df")

    for col in shared_columns:
        col_left = f"{col}_{label_left}"
        col_right = f"{col}_{label_right}"

        def resolve(row: pd.Series) -> str:
            raw_left = row.get(col_left)
            raw_right = row.get(col_right)
            val_left = "" if pd.isna(raw_left) or str(
                raw_left).lower() == "null" else str(raw_left)
            val_right = "" if pd.isna(raw_right) or str(
                raw_right).lower() == "null" else str(raw_right)
            attr_type = _infer_attr_type(col, mda_df)
            if attr_type in {"int", "float"}:
                n_left = _to_number(val_left)
                n_right = _to_number(val_right)
                if n_left is None and n_right is None:
                    return ""
                if n_left is not None and n_right is not None:
                    if attr_type == "int":
                        if int(round(n_left)) == int(round(n_right)):
                            return _format_number(n_left, "int")
                    else:
                        if abs(n_left - n_right) <= 1e-9:
                            return _format_number(n_left, "float")
                disp_right = _format_number(
                    n_right, "int" if attr_type == "int" else "float") if n_right is not None else val_right
                disp_left = _format_number(
                    n_left, "int" if attr_type == "int" else "float") if n_left is not None else val_left
                right_part = f"{disp_right} (As-Designed)" if disp_right else ""
                left_part = f"{disp_left} (As-Procured)" if disp_left else ""
                return f"{right_part}\n{left_part}".strip()
            if val_left == val_right:
                return val_left.strip()
            right_part = f"{val_right} (As-Designed)" if val_right else ""
            left_part = f"{val_left} (As-Procured)" if val_left else ""
            return f"{right_part}\n{left_part}".strip()

        merged[col] = merged.apply(resolve, axis=1)
        merged.drop(columns=[col_left, col_right], inplace=True)

    mda_df_safe = st.session_state.get("mda_df", pd.DataFrame())
    merged = apply_electrical_inheritance(merged, mda_df_safe)
    merged = merged[merged[on].astype(str).str.endswith("RA-")]
    stats = merged["_merge"].value_counts(
        dropna=False).to_dict() if "_merge" in merged.columns else {}
    log(
        f"Smart merge completed | rows={len(merged)} | unique {on}={merged[on].nunique()} | match_stats={stats}",
        level="INFO",
    )
    merged = apply_attribute_order(merged, mda_df_safe)
    return merged


@st.cache_data
def read_inputs(uploaded_files) -> Dict[str, pd.DataFrame]:
    dfs: Dict[str, pd.DataFrame] = {}
    for file in uploaded_files or []:
        try:
            df = pd.read_csv(file)
        except Exception as exc:
            st.error(f"Error reading {getattr(file, 'name', 'file')}: {exc}")
            log(
                f"Exception while reading {getattr(file, 'name', 'file')}: {exc}", level="ERROR")
            continue
        if "RepeFonct" not in df.columns:
            st.error(
                f"File `{file.name}` does not contain 'RepeFonct' column.")
            log(
                f"'RepeFonct' column missing in file: {file.name}. Columns={list(df.columns)}",
                level="WARNING",
            )
            continue
        dfs[file.name] = df
    st.session_state["dfs"] = dfs
    return dfs


def merge_df(dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    items = list(dfs.items())
    if not items:
        matched = pd.DataFrame()
        st.session_state["matched_df"] = matched
        st.session_state["unmatched_df"] = pd.DataFrame()
        st.session_state["merged_df"] = matched
        st.session_state["merge_sources"] = {}
        st.session_state["grid_data_df"] = None
        st.session_state["grid_filters"] = {}
        st.session_state["grid_search"] = ""
        st.session_state["grid_export_bytes"] = None
        return matched

    if len(items) == 1:
        name, df = items[0]
        matched = apply_attribute_order(
            df.copy(), st.session_state.get("mda_df", pd.DataFrame()))
        st.session_state["matched_df"] = matched
        st.session_state["unmatched_df"] = pd.DataFrame()
        st.session_state["merged_df"] = matched
        st.session_state["merge_sources"] = {"left": name, "right": None}
        st.session_state["grid_data_df"] = None
        st.session_state["grid_filters"] = {}
        st.session_state["grid_search"] = ""
        st.session_state["grid_export_bytes"] = None
        return matched

    if len(items) > 2:
        log(
            "More than two input files provided. Matched/unmatched split applies to the first two inputs only.",
            level="WARNING",
        )

    (left_name, df_left), (right_name, df_right) = items[0], items[1]
    st.session_state["merge_sources"] = {
        "left": left_name, "right": right_name}

    merged = smart_merge(
        df_left,
        df_right,
        on="RepeFonct",
        label_left=_slugify(left_name),
        label_right=_slugify(right_name),
    )

    left_only = right_only = 0
    if "_merge" in merged.columns:
        counts = merged["_merge"].value_counts()
        left_only = int(counts.get("left_only", 0))
        right_only = int(counts.get("right_only", 0))
        matched = merged[merged["_merge"] == "both"].copy()
        unmatched = merged[merged["_merge"] != "both"].copy()
        source_map = {"left_only": left_name, "right_only": right_name}
        if not unmatched.empty:
            unmatched["Source"] = unmatched["_merge"].map(
                source_map).fillna("unknown")
        matched.drop(columns=["_merge"], inplace=True, errors="ignore")
        unmatched.drop(columns=["_merge"], inplace=True, errors="ignore")
    else:
        matched = merged.copy()
        unmatched = pd.DataFrame()

    matched = matched.reset_index(drop=True)
    if not unmatched.empty:
        unmatched = unmatched.reset_index(drop=True)

    st.session_state["matched_df"] = matched
    st.session_state["unmatched_df"] = unmatched
    st.session_state["merged_df"] = matched
    st.session_state["grid_data_df"] = None
    st.session_state["grid_filters"] = {}
    st.session_state["grid_search"] = ""
    st.session_state["grid_export_bytes"] = None

    log(
        "Matched rows: {matched_cnt} | Unmatched rows: {unmatched_cnt} "
        "(left_only={left_only}, right_only={right_only})".format(
            matched_cnt=len(matched),
            unmatched_cnt=len(unmatched),
            left_only=left_only,
            right_only=right_only,
        ),
        level="INFO",
    )

    return matched


def show_n_select_ecs(dfs: Dict[str, pd.DataFrame]):
    cols = st.columns([0.75, 0.25], gap="small")
    with cols[0]:
        st.subheader("Search Results")
        progress = st.progress(0, text="Preparing data...")
        selected = None
        if "merged_df" not in st.session_state:
            st.session_state["merged_df"] = None
        if st.session_state["merged_df"] is None:
            merged_df = merge_df(dfs)
            progress.progress(20, text="Merging inputs...")
        else:
            merged_df = st.session_state["merged_df"]
            progress.progress(20, text="Using cached merged data...")

        search = st.sidebar.text_input("Search", label_visibility="visible")
        current_search = search or ""

        _pre = merged_df.copy()
        if "RepeFonct" in _pre.columns:
            _pre["CodMatExt"] = _pre["RepeFonct"].astype(str).str.slice(8, 11)

        curr_filters: Dict[str, tuple] = {}
        label_map = _attribute_label_map(st.session_state.get("mda_df"))
        try:
            norm_map = {re.sub(r"\W+", "", str(k)).lower()                        : v for k, v in label_map.items()}
        except Exception:
            norm_map = {}

        with st.sidebar.expander("Filters", expanded=True):
            def _apply_multiselect(df: pd.DataFrame, col: str) -> pd.DataFrame:
                if col not in df.columns:
                    curr_filters[col] = tuple()
                    return df
                options = df[col].astype(str).dropna().replace(
                    {"nan": ""}).unique().tolist()
                options = sorted([opt for opt in options if opt != ""])
                display_label = label_map.get(col)
                if not display_label:
                    key = re.sub(r"\W+", "", str(col)).lower()
                    display_label = norm_map.get(key, col)
                selection = st.multiselect(
                    display_label,
                    options,
                    key=f"flt_{col}")
                curr_filters[col] = tuple(selection)
                return df

            for col in ("ElemSys", "Component Kind", "ComponentKind", "Contrat", "CodMatExt", "SGApp"):
                _pre = _apply_multiselect(_pre, col)

        prev_filters = st.session_state.get("grid_filters", {})
        prev_search = st.session_state.get("grid_search", "")
        need_compute = (
            st.session_state.get("grid_data_df") is None
            or prev_filters != curr_filters
            or prev_search != current_search
        )

        if need_compute:
            progress.progress(45, text="Applying filters...")
            base_df = st.session_state.get("merged_df").copy()
            data_df = base_df
            if "RepeFonct" in data_df.columns and "CodMatExt" not in data_df.columns:
                data_df["CodMatExt"] = data_df["RepeFonct"].astype(
                    str).str.slice(8, 11)
            for pattern in _search_patterns(current_search):
                try:
                    mask = data_df.apply(
                        lambda row: row.astype(str).str.contains(
                            pattern, case=False, regex=True, na=False).any(),
                        axis=1,
                    )
                    data_df = data_df[mask]
                except Exception:
                    continue
            for col, values in curr_filters.items():
                if col in data_df.columns and values:
                    data_df = data_df[data_df[col].astype(
                        str).isin(list(values))]
            if "::auto_unique_id::" in data_df.columns:
                data_df = data_df.drop(columns="::auto_unique_id::")
            progress.progress(65, text="Preparing grid...")
            st.session_state["grid_data_df"] = data_df.reset_index(drop=True)
            st.session_state["grid_filters"] = curr_filters.copy()
            st.session_state["grid_search"] = current_search
            st.session_state["grid_export_bytes"] = None
        else:
            data_df = st.session_state.get("grid_data_df")
            if data_df is None:
                data_df = st.session_state.get("merged_df").copy()
            if "::auto_unique_id::" in data_df.columns:
                data_df = data_df.drop(columns="::auto_unique_id::")
            progress.empty()

        st.sidebar.caption(f"{len(data_df)} results found")

        try:
            if need_compute or st.session_state.get("grid_export_bytes") is None:
                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
                    data_df.to_excel(writer, index=False,
                                     sheet_name="filtered")
                buffer.seek(0)
                st.session_state["grid_export_bytes"] = buffer.getvalue()
            export_bytes = st.session_state.get("grid_export_bytes")
            if export_bytes:
                st.download_button(
                    label="Export filtered (XLSX)",
                    data=export_bytes,
                    file_name="filtered_results.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    type="secondary",
                    help="Download the filtered table as Excel",
                    key="dl_filtered_xlsx",
                )
        except Exception:
            pass

        gb = GridOptionsBuilder.from_dataframe(data_df)
        try:
            if label_map:
                for col in data_df.columns:
                    if col in label_map and label_map[col] != col:
                        gb.configure_column(col, headerName=label_map[col])
        except Exception:
            pass
        try:
            if len(data_df.columns) > 0:
                first_col = "RepeFonct" if "RepeFonct" in data_df.columns else data_df.columns[
                    0]
                gb.configure_column(first_col, pinned="left")
        except Exception:
            pass
        gb.configure_selection("single", use_checkbox=False)
        gb.configure_pagination(paginationAutoPageSize=False)
        gb.configure_default_column(editable=False, groupable=True)
        grid_options = gb.build()
        grid_options["paginationPageSize"] = 100

        if need_compute:
            progress.progress(90, text="Rendering grid...")

        grid_response = AgGrid(
            data_df,
            gridOptions=grid_options,
            update_mode="SELECTION_CHANGED",
            fit_columns_on_grid_load=False,
            theme="alpine",
            custom_css={
                ".ag-header-cell-label": {"font-size": "11px !important"},
                ".ag-cell": {"font-size": "11px !important", "line-height": "18px !important"},
            },
        )

        if need_compute:
            progress.progress(100, text="Done")
            progress.empty()

        selected = grid_response["selected_rows"]

    if selected is not None:
        selected_df = pd.DataFrame(selected).reset_index(drop=True)
        st.session_state["selected_ecs"] = True
        st.session_state["selected_ecs_df"] = selected_df
        with cols[1]:
            ecs = selected_df.loc[0,
                                  "RepeFonct"] if "RepeFonct" in selected_df.columns else ""
            st.subheader(str(ecs))
            if st.button("Export STF", type="primary"):
                try:
                    out_path = export_stf_from_row(selected_df)
                    st.success(f"Export success: {out_path}")
                except Exception as exc:
                    st.error(f"Export failed: {exc}")
            id_card_columns(selected_df)
    else:
        st.session_state["selected_ecs"] = False


def apply_electrical_inheritance(merged: pd.DataFrame, mda_df: pd.DataFrame) -> pd.DataFrame:
    """Propagate electrical attributes from M-source ECS when RepeFonct ends with RA-."""
    if merged is None or merged.empty or mda_df is None or mda_df.empty:
        return merged
    if "Facet" not in mda_df.columns or "MUDU" not in mda_df.columns:
        return merged

    electrical_attrs = mda_df.loc[mda_df["Facet"]
                                  == "2_Electrical", "MUDU"].tolist()
    electrical_attrs = [
        attr for attr in electrical_attrs if attr in merged.columns]
    if not electrical_attrs:
        return merged

    source = merged[["RepeFonct"] + electrical_attrs].copy()
    source.rename(
        columns={attr: f"{attr}_Msrc" for attr in electrical_attrs}, inplace=True)
    source.rename(columns={"RepeFonct": "RepeFonct_Msrc"}, inplace=True)

    merged = merged.merge(
        source,
        how="left",
        left_on="ECS_Mtr",
        right_on="RepeFonct_Msrc",
    )

    ra_mask = merged["RepeFonct"].astype(str).str.endswith("RA-")

    for attr in electrical_attrs:
        current = merged.loc[ra_mask, attr].astype(str)
        to_replace = (
            current.str.contains(r"\[ne\]", case=False, na=True)
            | current.str.strip().str.lower().isin({"", "-", "nan", "none", "null"})
        )
        merged.loc[ra_mask & to_replace,
                   attr] = merged.loc[ra_mask & to_replace, f"{attr}_Msrc"]

    merged.drop(columns=["RepeFonct_Msrc"] +
                [f"{attr}_Msrc" for attr in electrical_attrs], inplace=True)
    return merged
