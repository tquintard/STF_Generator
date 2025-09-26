import streamlit as st
import pandas as pd


def id_card_columns(selected):
    ecs = selected.loc[0, "RepeFonct"]
    st.subheader(f"🆔{ecs}")
    mda_df, facet_labels = st.session_state['mda_df'], st.session_state['facet_labels']
    # Création d'autant de colonnes que de facettes
    cols = st.columns(len(facet_labels))

    # Remplir chaque colonne avec les attributs de la facette
    for col, facet in zip(cols, facet_labels):
        with col:
            # Si un label existe pour cette facette, on l’utilise, sinon on garde le code
            label = facet_labels.get(facet, facet)
            with st.expander(label, expanded=True):
                group = mda_df[mda_df["Facet"] == facet]
                for _, row in group.iterrows():
                    attr = row["MUDU"]
                    try:
                        value = selected.loc[0, attr]
                        st.markdown(
                            f"**{attr}:** {value if pd.notna(value) else '-'}",
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

    if not st.session_state.inputs_integrated:
        st.sidebar.subheader("⬇️ Upload Inputs")
        uploaded_files = st.sidebar.file_uploader(
            "Sélectionne tes inputs",
            type=["csv"],
            accept_multiple_files=True
        )
        if uploaded_files:
            st.session_state.uploaded_files = uploaded_files

        if st.sidebar.button("Integrate"):
            if st.session_state.uploaded_files:
                dfs = read_inputs(uploaded_files)
                st.session_state.dataframes = dfs
                st.session_state.inputs_integrated = True
                st.success("Inputs intégrés avec succès ✅")
                st.rerun()
    else:
        st.sidebar.subheader("⬇️ Upload Inputs")
        st.sidebar.write("Inputs intégrés en cache.")
        if st.sidebar.button("Reset"):
            st.session_state.inputs_integrated = False
            st.session_state.uploaded_files = None
            st.session_state.dataframes = None
            st.rerun()
