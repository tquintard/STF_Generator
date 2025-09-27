import streamlit as st
import pandas as pd


def id_card_columns(selected):
    mda_df, facet_labels = st.session_state['mda_df'], st.session_state['facet_labels']
    # Map MUDU -> Label (fallback to Name when missing)
    label_map = {}
    try:
        label_col = None
        for c in ("Label", "Display", "Libelle", "Libellé"):
            if c in mda_df.columns:
                label_col = c
                break
        if label_col:
            tmp = mda_df[["MUDU", label_col]].copy()
            tmp = tmp.dropna(subset=["MUDU"])  # keep rows with defined MUDU
            for _, r in tmp.iterrows():
                mudu = str(r["MUDU"])
                lab = r[label_col]
                if pd.notna(lab) and str(lab).strip() != "":
                    label_map[mudu] = str(lab)
    except Exception:
        label_map = {}

    # Remplir chaque colonne avec les attributs de la facette
    for facet in facet_labels:
        # Si un label existe pour cette facette, on l’utilise, sinon on garde le code
        label = facet_labels.get(facet, facet)
        with st.expander(label, expanded=True):
            group = mda_df[mda_df["Facet"] == facet]
            # Ordonner les attributs par la colonne 'Order' si disponible
            if "Order" in group.columns:
                group = group.copy()
                group["__OrderNum__"] = pd.to_numeric(
                    group["Order"], errors="coerce")
                group = group.sort_values(
                    ["__OrderNum__", "MUDU"], kind="mergesort")
            for _, row in group.iterrows():
                attr = row["MUDU"]
                try:
                    value = selected.loc[0, attr]
                    display = label_map.get(str(attr), str(attr))
                    st.markdown(
                        f"**{display}:** {value if pd.notna(value) else '-'}",
                    )
                except KeyError:
                    pass


def upload_sidebar():
    # --- Sidebar section ---
    input_uploader_expanded = True if 'inputs_uploaded' in st.session_state is None else False
    with st.sidebar.expander("⬆️ Upload Inputs", expanded=input_uploader_expanded):
        st.session_state['inputs_uploaded'] = st.file_uploader(
            "Upload input CSV files:",
            type=["csv"],
            accept_multiple_files=True,
            label_visibility="hidden",
        )
        return st.session_state["inputs_uploaded"]

# Cache la lecture CSV


@st.cache_data
def load_csv(file):
    return pd.read_csv(file)


def sidebar_inputs():
    from modules.utils import read_inputs

    """Gestion de l’upload et intégration des inputs dans la sidebar."""

    if "inputs_integrated" not in st.session_state:
        st.session_state.inputs_integrated = False
        st.session_state.uploaded_files = None
        st.session_state.dataframes = None

    with st.sidebar.expander("Upload Inputs", icon="⬇️", expanded=True):
        if not st.session_state.inputs_integrated:
            uploaded_files = st.file_uploader(
                "Sélectionne tes inputs",
                type=["csv"],
                accept_multiple_files=True
            )
            if uploaded_files:
                st.session_state.uploaded_files = uploaded_files

            if st.button("Integrate"):
                if st.session_state.uploaded_files:
                    dfs = read_inputs(uploaded_files)
                    st.session_state.dataframes = dfs
                    st.session_state.inputs_integrated = True
                    st.success("Inputs intégrés avec succès")
                    st.rerun()
        else:
            st.success("Inputs intégrés avec succès")
            if st.button("Reset"):
                st.session_state.inputs_integrated = False
                st.session_state.uploaded_files = None
                st.session_state.dataframes = None
                st.session_state.merged_df = None
                st.rerun()
