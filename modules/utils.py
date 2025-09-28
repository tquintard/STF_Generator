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

from modules.exporter import export_stf_from_row, export_stf_batch
from modules.ui_modules import id_card_columns
from mdm.mdm_loader import attribute_label_map, apply_attribute_order, infer_attr_type


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


def _safe_dirname(name: str, default: str) -> str:
    value = str(name).strip() if name is not None else ""
    if not value:
        value = default
    sanitized = re.sub(r"[^0-9A-Za-z_-]+", "_", value)
    return sanitized or default


def _search_patterns(query: str) -> list[str]:
    """Split search query on commas and convert * wildcards to regex patterns."""
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
            attr_type = infer_attr_type(col, mda_df)
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
    if "ecs_tabs" not in st.session_state:
        st.session_state["ecs_tabs"] = []
    if "ecs_skip_next_add" not in st.session_state:
        st.session_state["ecs_skip_next_add"] = None
    if "ecs_grid_key" not in st.session_state:
        st.session_state["ecs_grid_key"] = 0
    if "merged_df" not in st.session_state:
        st.session_state["merged_df"] = None

    existing_tabs_snapshot = list(st.session_state["ecs_tabs"])
    tab_labels = ["Search"] + [tab["label"] for tab in existing_tabs_snapshot]
    tabs = st.tabs(tab_labels or ["Search"])
    selected_rows = None
    export_bytes = None

    with tabs[0]:
        st.subheader("Search Results")
        status_placeholder = st.empty()
        status_active = False

        if st.session_state["merged_df"] is None:
            status_active = True
            status_placeholder.caption("Preparing data...")
            merged_df = merge_df(dfs)
            status_placeholder.caption("Merging inputs...")
        else:
            merged_df = st.session_state["merged_df"]

        search = st.sidebar.text_input("Search", label_visibility="visible")
        current_search = search or ""

        _pre = merged_df.copy()
        if "RepeFonct" in _pre.columns:
            _pre["CodMatExt"] = _pre["RepeFonct"].astype(str).str[-3:]

        curr_filters: Dict[str, tuple] = {}
        label_map = attribute_label_map(st.session_state.get("mda_df"))
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
                    display_label, options, key=f"flt_{col}")
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
            status_active = True
            status_placeholder.caption("Applying filters...")
            data_df = st.session_state.get("merged_df").copy()
            if "RepeFonct" in data_df.columns and "CodMatExt" not in data_df.columns:
                data_df["CodMatExt"] = data_df["RepeFonct"].astype(
                    str).str[-3:]
            patterns = _search_patterns(current_search)
            for pattern in patterns:
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
            status_placeholder.caption("Preparing grid...")
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
            status_placeholder.empty()
            status_active = False

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
        except Exception:
            export_bytes = None

        gb = GridOptionsBuilder.from_dataframe(data_df)
        label_map = attribute_label_map(st.session_state.get("mda_df"))
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

        if status_active:
            status_placeholder.caption("Rendering grid...")

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
            key=f"ecs_grid_{st.session_state['ecs_grid_key']}",
        )

        if status_active:
            status_placeholder.caption("Done")

        selected_rows = grid_response["selected_rows"]

        action_cols = st.columns([1, 1])
        with action_cols[0]:
            if st.button("Export all STF (filtered)", key="export_all_stf"):
                try:
                    batch_path = export_stf_batch(data_df)
                    if batch_path:
                        st.success(f"Batch STF exported: {batch_path}")
                    else:
                        st.info("No STF generated (no matching templates).")
                except Exception as exc:
                    st.error(f"Batch export failed: {exc}")
                    log(f"Batch export failed: {exc}", level="ERROR")
        with action_cols[1]:
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

    tabs_state = st.session_state["ecs_tabs"]
    skip_next_id = st.session_state.get("ecs_skip_next_add")
    if selected_rows is not None and len(selected_rows) > 0:
        selected_df = pd.DataFrame(selected_rows).reset_index(drop=True)
        ecs_value = ""
        if not selected_df.empty and "RepeFonct" in selected_df.columns:
            ecs_value = str(selected_df.loc[0, "RepeFonct"]).strip()
        label = ecs_value or "ECS"
        tab_id = ecs_value or f"ecs_{len(tabs_state) + 1}"
        existing_idx = next((idx for idx, tab in enumerate(
            tabs_state) if tab.get("id") == tab_id), None)
        if existing_idx is None:
            if skip_next_id == tab_id:
                st.session_state["ecs_skip_next_add"] = None
            else:
                base_key = f"ecs_{_slugify(label)}" if _slugify(
                    label) else f"ecs_{len(tabs_state) + 1}"
                key = base_key
                suffix = 1
                existing_keys = {tab["key"] for tab in tabs_state}
                while key in existing_keys:
                    suffix += 1
                    key = f"{base_key}_{suffix}"
                tabs_state.append(
                    {"id": tab_id, "key": key, "label": label, "data": selected_df})
                st.session_state["ecs_tabs"] = tabs_state
                st.session_state["selected_ecs_df"] = selected_df
                st.session_state["ecs_skip_next_add"] = None
                st.session_state["ecs_grid_key"] += 1
                st.rerun()
        else:
            tabs_state[existing_idx]["data"] = selected_df
            st.session_state["ecs_tabs"] = tabs_state
            st.session_state["selected_ecs_df"] = selected_df
            st.session_state["ecs_skip_next_add"] = None
    else:
        st.session_state["ecs_skip_next_add"] = None
    existing_tabs_snapshot = list(st.session_state["ecs_tabs"])
    st.session_state["selected_ecs"] = bool(existing_tabs_snapshot)

    for idx, tab_info in enumerate(existing_tabs_snapshot, start=1):
        with tabs[idx]:
            control_cols = st.columns([0.15, 0.1, 0.55, 0.02], gap="small")
            with control_cols[0]:
                st.subheader(tab_info["label"])
            with control_cols[1]:
                if st.button("Export STF", key=f"export_tab_{tab_info['key']}", type="primary"):
                    with control_cols[2]:
                        try:
                            out_path = export_stf_from_row(tab_info["data"])
                            st.success(f"Export success: {out_path}")
                        except Exception as exc:
                            st.error(f"Export failed: {exc}")

            with control_cols[3]:
                if st.button("❌", key=f"close_tab_{tab_info['key']}", type="tertiary"):
                    st.session_state["ecs_tabs"] = [
                        tab for tab in st.session_state["ecs_tabs"] if tab["key"] != tab_info["key"]]
                    st.session_state["ecs_skip_next_add"] = tab_info["id"]
                    if not st.session_state["ecs_tabs"]:
                        st.session_state["selected_ecs"] = False
                        st.session_state["selected_ecs_df"] = pd.DataFrame()
                    st.session_state["ecs_grid_key"] += 1
                    st.rerun()

            id_card_columns(tab_info["data"])


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
