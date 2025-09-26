# modules/utils.py

import os
import pandas as pd
import streamlit as st
from config.settings import log
from st_aggrid import AgGrid, GridOptionsBuilder
from modules.ui_modules import id_card_columns


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
            candidate = s[:-1] + "M"  # transforme RA- → RAM
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
            val_left = str(row[col_left]) if pd.notna(
                row[col_left]) and row[col_left] != "null" else ""
            val_right = str(row[col_right]) if pd.notna(
                row[col_right]) and row[col_right] != "null" else ""
            if val_left == val_right:
                return val_left
            val_right = f"{val_right} (As-Designed)" if val_right != "" else ""
            val_left = f"{val_left} (As-Procured)" if val_left != "" else ""
            return f"{val_right}\n{val_left}"

        merged[col] = merged.apply(resolve, axis=1)
        merged.drop(columns=[col_left, col_right], inplace=True)

    # Appliquer la règle d’héritage électrique après la résolution
    merged = apply_electrical_inheritance(merged, st.session_state["mda_df"])
    # Apply filter to keep only damper ECS (RA-)
    merged = merged[merged["RepeFonct"].str.endswith("RA-")]
    log(
        f"Smart merge completed | rows={len(merged)} | unique {on}={merged[on].nunique()}",
        level="INFO",
    )
    return merged


@st.cache_data
def read_inputs(uploaded_files):
    dfs = {}
    for file in uploaded_files:
        try:
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
        except Exception as e:
            st.error(f"Error reading {file.name}: {e}")
            log(f"Exception while reading {file.name}: {e}", level="ERROR")

    st.session_state["dfs"] = dfs
    return dfs


def merge_df(dfs):
    # --- Merge step ---
    merged_df = None
    for _, df in dfs.items():

        if merged_df is None:
            merged_df = df
        else:
            merged_df = smart_merge(
                merged_df, df, on="RepeFonct"
            )

    # ✅ Vérifier que la DF existe et n'est pas vide
    if merged_df is not None and not merged_df.empty:
        # Search bar
        search = st.text_input(
            "🔎 Search", placeholder="🔎 Search", label_visibility="hidden")

        if search is not None:
            mask = merged_df.apply(
                lambda row: row.astype(str).str.contains(search, case=False).any(), axis=1
            )
            filtered_df = merged_df[mask]
            st.sidebar.caption(f"{len(filtered_df)} résultats trouvés")
        else:
            filtered_df = merged_df
    else:
        filtered_df = pd.DataFrame()

    return filtered_df


def show_n_select_ecs(dfs):
    with st.expander("🔎 Search", expanded=True):

        filtered_df = merge_df(dfs)

        gb = GridOptionsBuilder.from_dataframe(filtered_df)
        # 'multiple' possible
        gb.configure_selection("single", use_checkbox=False)
        gb.configure_pagination(paginationAutoPageSize=False)
        gb.configure_default_column(editable=False, groupable=True)
        grid_options = gb.build()
        grid_options["paginationPageSize"] = 20
        grid_response = AgGrid(
            filtered_df,
            gridOptions=grid_options,
            update_mode="SELECTION_CHANGED",
            fit_columns_on_grid_load=False,
            theme="alpine",
        )
        selected = grid_response["selected_rows"]

    # === Identity Card ===
    if selected is not None:
        id_card_columns(selected.reset_index(drop=True))


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

    # 2. Préparer la table source des valeurs à hériter
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

        # valeurs vides / nullish / contient [NE] (même avec suffixes)
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
