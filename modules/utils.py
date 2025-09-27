# modules/utils.py

import os
import io
import pandas as pd
import streamlit as st
from config.settings import log
from st_aggrid import AgGrid, GridOptionsBuilder
from modules.exporter import export_stf_from_row
from modules.ui_modules import id_card_columns


def _attribute_label_map(mda_df: pd.DataFrame) -> dict:
    """Return mapping MUDU -> label from MDA if available.

    Tries common label columns: 'Label', 'Display', 'Libelle', 'Libellé'.
    """
    if mda_df is None or mda_df.empty or "MUDU" not in mda_df.columns:
        return {}
    label_col = None
    for c in ("Label", "Display", "Libelle", "Libellé"):
        if c in mda_df.columns:
            label_col = c
            break
    if label_col is None:
        return {}
    mapping = {}
    for _, row in mda_df.iterrows():
        mudu = row.get("MUDU")
        if pd.isna(mudu):
            continue
        lab = row.get(label_col)
        if pd.notna(lab) and str(lab).strip() != "":
            mapping[str(mudu)] = str(lab)
    return mapping


def _infer_attr_type(attr: str, mda_df: pd.DataFrame) -> str:
    """Infer attribute type from MDA. Returns one of: 'int', 'float', 'str'."""
    try:
        if mda_df is None or mda_df.empty or "MUDU" not in mda_df.columns:
            return "str"
        row = mda_df.loc[mda_df["MUDU"] == attr]
        if row.empty:
            return "str"
        # Try common column names for type
        type_col = None
        for c in ["Type", "AttrType", "DataType", "AttributeType", "ColumnType"]:
            if c in mda_df.columns:
                type_col = c
                break
        if type_col is None:
            return "str"
        t = str(row.iloc[0][type_col]).strip().lower()
        if any(x in t for x in ["int", "integer"]):
            return "int"
        if any(x in t for x in ["float", "double", "decimal", "number", "numeric", "real"]):
            return "float"
        return "str"
    except Exception:
        return "str"


def _to_number(val):
    """Parse a value to a float if possible, handling FR/EN formats.
    Returns float or None.
    """
    if val is None:
        return None
    s = str(val).strip()
    if s == "" or s.lower() in {"nan", "none", "null", "-"}:
        return None
    # Remove spaces and apostrophes used as thousand separators
    s = s.replace("\u00A0", " ").replace(" ", "").replace("'", "")
    # If both '.' and ',' exist, assume '.' thousand sep and ',' decimal
    if "," in s and "." in s:
        s = s.replace(".", "").replace(",", ".")
    elif "," in s and "." not in s:
        s = s.replace(",", ".")
    try:
        return float(s)
    except Exception:
        return None


def _format_number(num: float, kind: str) -> str:
    """Format a number consistently for display (int or float)."""
    if num is None:
        return ""
    if kind == "int":
        try:
            return str(int(round(num)))
        except Exception:
            return str(num)
    # float formatting: minimal, no trailing zeros, up to 6 decimals
    s = (f"{num:.6f}").rstrip("0").rstrip(".")
    return s


def apply_attribute_order(df: pd.DataFrame, mda_df: pd.DataFrame) -> pd.DataFrame:
    """
    Reorder DataFrame columns to respect the attribute order defined in the
    MDM (sheet "Attribute"), using column "Order" and attribute id column "MUDU".

    Rules:
    - Keep `RepeFonct` first when present.
    - Keep `ECS_Mtr` second when present.
    - Then, order attributes by ascending `Order` from the MDM.
    - Any remaining columns (not in MDM) are appended preserving current order.
    - If MDM or required columns are missing, returns df unchanged.
    """
    if df is None or df.empty:
        return df

    if mda_df is None or mda_df.empty:
        return df

    if not {"MUDU", "Order"}.issubset(mda_df.columns):
        return df

    all_cols = list(df.columns)

    # Filter MDM attributes that exist in the DataFrame
    mda_sub = mda_df[mda_df["MUDU"].isin(all_cols)].copy()
    if mda_sub.empty:
        return df

    # Ensure numeric ordering for robust sorting
    mda_sub["__OrderNum__"] = pd.to_numeric(mda_sub["Order"], errors="coerce")
    mda_sub.sort_values(["__OrderNum__", "MUDU"],
                        inplace=True, kind="mergesort")

    ordered_attrs = mda_sub["MUDU"].tolist()

    new_cols = []
    if "RepeFonct" in all_cols:
        new_cols.append("RepeFonct")

    # Add attributes in requested order, skipping ones already placed
    for attr in ordered_attrs:
        if attr in df.columns and attr not in new_cols:
            new_cols.append(attr)

    # Append remaining columns preserving their current relative order
    remaining = [c for c in all_cols if c not in new_cols]
    new_cols.extend(remaining)

    return df[new_cols]


def smart_merge(df_left, df_right, on="RepeFonct", label_left="1D", label_right="2D"):
    """
    Merge two DataFrames on a key, handling shared columns with custom logic.

    If a column exists in both DataFrames (other than the key):
      - If values are identical, keep the value
      - If values differ, concatenate:
        '{val_right} (As-Designed)\n{val_left} (As-Procured)'
    """
    merged = pd.merge(
        df_left, df_right, on=on, how="outer",
        suffixes=(f"_{label_left}", f"_{label_right}")
    )

    # Nouvelle définition de ECS_Mtr :
    existing_repefonct = set(merged["RepeFonct"].astype(str))

    def compute_ecs_mtr(x):
        s = str(x)
        if s.endswith("RA-"):
            candidate = s[:-1] + "M"  # transforme RA- â†’ RAM
            if candidate in existing_repefonct:
                return candidate
        return None

    merged["ECS_Mtr"] = merged["RepeFonct"].apply(compute_ecs_mtr)

    # Résolution des colonnes communes (fusion 1D/2D)
    shared_columns = set(df_left.columns).intersection(df_right.columns) - {on}
    for col in shared_columns:
        col_left = f"{col}_{label_left}"
        col_right = f"{col}_{label_right}"

        def resolve(row):
            """
            Resolve a shared column between two DataFrames by applying a custom logic.

            The logic is as follows:

            - If the values are identical, keep the value
            - If the values differ, concatenate the values with a newline separator,
            and add a label indicating the origin of the value (As-Designed or As-Procured)

            Args:
                row (pandas.Series): The row of the DataFrame to resolve

            Returns:
                str: The resolved value
            """
            raw_left = row[col_left]
            raw_right = row[col_right]

            # Normalize nullish
            val_left = "" if pd.isna(raw_left) or str(
                raw_left).lower() == "null" else str(raw_left)
            val_right = "" if pd.isna(raw_right) or str(
                raw_right).lower() == "null" else str(raw_right)

            # Type-aware comparison using MDA
            attr_type = _infer_attr_type(col, st.session_state.get("mda_df"))

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
                right_part = f"{disp_right} (As-Designed)" if disp_right != "" else ""
                left_part = f"{disp_left} (As-Procured)" if disp_left != "" else ""
                return f"{right_part}\n{left_part}".strip()
            else:
                if val_left == val_right:
                    return val_left.strip()
                right_part = f"{val_right} (As-Designed)" if val_right != "" else ""
                left_part = f"{val_left} (As-Procured)" if val_left != "" else ""
                return f"{right_part}\n{left_part}".strip()

        merged[col] = merged.apply(resolve, axis=1)
        merged.drop(columns=[col_left, col_right], inplace=True)

    # Appliquer la rÃ¨gle dâ€™héritage électrique aprÃ¨s la résolution
    merged = apply_electrical_inheritance(merged, st.session_state["mda_df"])
    # Apply filter to keep only damper ECS (RA-)
    merged = merged[merged["RepeFonct"].str.endswith("RA-")]
    log(
        f"Smart merge completed | rows={len(merged)} | unique {on}={merged[on].nunique()}",
        level="INFO",
    )
    merged = apply_attribute_order(merged, st.session_state["mda_df"])
    st.session_state["merged_df"] = merged
    return merged


@st.cache_data
def read_inputs(uploaded_files):
    dfs = {}
    for file in uploaded_files:
        df = pd.read_csv(file)

        if "RepeFonct" not in df.columns:
            st.error(
                f"File `{file.name}` does not contain 'RepeFonct' column.")
            log(
                f"'RepeFonct' column missing in file: {file.name}. Columns={list(df.columns)}",
                level="WARNING",
            )
        else:
            dfs[file.name] = df

    st.session_state["dfs"] = dfs
    return dfs


def merge_df(dfs):
    # --- Merge step ---
    _merged_df = None
    for _, df in dfs.items():

        if _merged_df is None:
            _merged_df = df
        else:
            _merged_df = smart_merge(
                _merged_df, df, on="RepeFonct"
            )

    return _merged_df


def show_n_select_ecs(dfs):

    cols = st.columns([0.75, 0.25], gap="small")
    with cols[0]:
        st.subheader("🔎 Search Results")
        progress = st.progress(0, text="Preparing dataâ€¦")
        selected = None
        if "merged_df" not in st.session_state:
            st.session_state["merged_df"] = None
        if st.session_state["merged_df"] is None:
            merged_df = merge_df(dfs)
            progress.progress(20, text="Merging inputsâ€¦")
        else:
            merged_df = st.session_state["merged_df"]
            progress.progress(20, text="Using cached merged dataâ€¦")

        # Search bar
        search = st.sidebar.text_input(
            "🔎 Search", label_visibility="visible")

        if False:
            mask = merged_df.apply(
                lambda row: row.astype(str).str.contains(search, case=False).any(), axis=1
            )
            merged_df = merged_df[mask]
        # Progress will be updated after filters are applied (only when needed)

        # Default sidebar filters on key MUDUs
        # Build a working copy and apply filters before free-text search
        _pre = merged_df.copy()
        try:
            if "RepeFonct" in _pre.columns:
                _pre["CodMatExt"] = _pre["RepeFonct"].astype(
                    str).str.slice(0, 3)
        except Exception:
            pass

        with st.sidebar.expander("Filters", expanded=True):
            curr_filters = {}
            try:
                _label_map = _attribute_label_map(
                    st.session_state.get("mda_df")) or {}
            except Exception:
                _label_map = {}
            try:
                import re as _re
                _norm_map = {(_re.sub(r"\W+", "", str(k)).lower())                             : v for k, v in _label_map.items()}
            except Exception:
                _norm_map = {}

            def _apply_multiselect(df, col):
                if col in df.columns:
                    opts = (
                        df[col].astype(str).dropna().replace(
                            {"nan": ""}).unique().tolist()
                    )
                    # clean empties
                    opts = sorted([o for o in opts if o != ""])
                    _label = _label_map.get(col)
                    if not _label:
                        try:
                            _norm = _re.sub(r"\W+", "", str(col)).lower()
                            _label = _norm_map.get(_norm, col)
                        except Exception:
                            _label = col
                    sel = st.multiselect(_label, opts, key=f"flt_{col}")
                    curr_filters[col] = tuple(sel)
                return df

            for _col in ("ElemSys", "Component Kind", "ComponentKind", "Contrat", "CodMatExt"):
                _pre = _apply_multiselect(_pre, _col)

        # Apply the pre-filters to the dataset used by the search/grid
        merged_df = _pre
        progress.progress(45, text="Applying filtersâ€¦")

# replaced by cached caption below

        # Compute or reuse filtered data for grid (avoid recomputation on selection changes)
        prev_filters = st.session_state.get("grid_filters", {})
        prev_search = st.session_state.get("grid_search", "")
        need_compute = (
            st.session_state.get("grid_data_df") is None
            or prev_filters != curr_filters
            or prev_search != (search or "")
        )

        if not need_compute:
            try:
                progress.empty()
            except Exception:
                pass

        if need_compute:
            base_df = st.session_state.get("merged_df")
            data_df = base_df.copy()
            try:
                if "RepeFonct" in data_df.columns and "CodMatExt" not in data_df.columns:
                    data_df["CodMatExt"] = data_df["RepeFonct"].astype(
                        str).str.slice(0, 3)
            except Exception:
                pass
            if search:
                try:
                    mask = data_df.apply(
                        lambda row: row.astype(str).str.contains(search, case=False).any(), axis=1
                    )
                    data_df = data_df[mask]
                except Exception:
                    pass
            for col, vals in curr_filters.items():
                if col in data_df.columns and vals:
                    data_df = data_df[data_df[col].astype(
                        str).isin(list(vals))]
            st.session_state["grid_data_df"] = data_df
            st.session_state["grid_filters"] = curr_filters
            st.session_state["grid_search"] = search or ""
        else:
            data_df = st.session_state.get("grid_data_df")

        st.sidebar.caption(f"{len(data_df)} résultats trouvés")

        # Export filtered results to Excel
        try:
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
                data_df.to_excel(writer, index=False, sheet_name="filtered")
            buffer.seek(0)
            st.download_button(
                label="Export filtered (XLSX)",
                data=buffer,
                file_name="filtered_results.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                type="secondary",
                help="Télécharger le tableau filtré au format Excel",
                key="dl_filtered_xlsx",
            )
        except Exception:
            pass

        gb = GridOptionsBuilder.from_dataframe(data_df)
        # Apply labels from MDA to column headers
        try:
            label_map = _attribute_label_map(st.session_state.get("mda_df"))
            if label_map:
                for c in data_df.columns:
                    if c in label_map and label_map[c] != c:
                        gb.configure_column(c, headerName=label_map[c])
        except Exception:
            pass
        # Pin first column on the left
        try:
            if len(data_df.columns) > 0:
                first_col = "RepeFonct" if "RepeFonct" in data_df.columns else data_df.columns[
                    0]
                gb.configure_column(first_col, pinned="left")
        except Exception:
            pass
        # 'multiple' possible
        gb.configure_selection("single", use_checkbox=False)
        gb.configure_pagination(paginationAutoPageSize=False)
        gb.configure_default_column(editable=False, groupable=True)

        grid_options = gb.build()
        grid_options["paginationPageSize"] = 100

        progress.progress(90, text="Rendering gridâ€¦")
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
        progress.progress(100, text="Done")
        progress.empty()
        selected = grid_response["selected_rows"]

    # === Identity Card ===
    if selected is not None:
        selected = selected.reset_index(drop=True)
        st.session_state["selected_ecs"] = True
        st.session_state["selected_ecs_df"] = selected
        with cols[1]:
            ecs = selected.loc[0, "RepeFonct"]
            st.subheader(f"🔩{ecs}")
            # Export section
            if st.button("Export STF", type="primary"):
                # try:
                out_path = export_stf_from_row(selected)
                st.success(f"Export réussi: {out_path}")
                # except Exception as e:
                #     st.error(f"Ã‰chec export: {e}")
            id_card_columns(selected)
    else:
        st.session_state["selected_ecs"] = False


def apply_electrical_inheritance(merged: pd.DataFrame, mda_df: pd.DataFrame) -> pd.DataFrame:
    """
    Pour tous les ECS dont RepeFonct finit par 'RA-', valoriser les attributs
    de la Facet '2_Electrical' avec les valeurs de l'ECS correspondant
    (ECS_Mtr).
    """
    if merged.empty or mda_df.empty:
        return merged

    if "Facet" not in mda_df or "MUDU" not in mda_df:
        return merged

    # 1. Liste des attributs électriques
    electrical_attrs = mda_df.loc[mda_df["Facet"]
                                  == "2_Electrical", "MUDU"].tolist()
    electrical_attrs = [
        col for col in electrical_attrs if col in merged.columns]

    if not electrical_attrs:
        return merged

    # 2. Préparer la table source des valeurs Ã  hériter
    source = merged[["RepeFonct"] + electrical_attrs].copy()
    source.rename(
        columns={attr: attr + "_Msrc" for attr in electrical_attrs}, inplace=True)
    source.rename(columns={"RepeFonct": "RepeFonct_Msrc"}, inplace=True)

    # 3. Merge pour récupérer les valeurs depuis ECS_Mtr
    merged = merged.merge(
        source,
        how="left",
        left_on="ECS_Mtr",
        right_on="RepeFonct_Msrc"
    )

    # 4. Appliquer les héritages si RepeFonct se termine par 'RA-'
    ra_mask = merged["RepeFonct"].astype(str).str.endswith("RA-")

    for attr in electrical_attrs:
        current_val = merged.loc[ra_mask, attr].astype(str)

        # valeurs vides / nullish / contient [NE] (mÃªme avec suffixes)
        to_replace = (
            current_val.str.contains(r"\[ne\]", case=False, na=True) |
            current_val.str.strip().str.lower().isin(
                {"", "-", "nan", "none", "null"})
        )

        inherited_val = merged.loc[ra_mask, attr + "_Msrc"]
        merged.loc[ra_mask & to_replace, attr] = inherited_val

    # 5. Nettoyage
    merged.drop(columns=["RepeFonct_Msrc"] +
                [a + "_Msrc" for a in electrical_attrs], inplace=True)

    return merged
