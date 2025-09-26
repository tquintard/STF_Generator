# Home.py

import streamlit as st
from config.settings import APP_NAME, APP_VERSION, MDM_FILE, log
from mdm.mdm_loader import load_mda
from modules.ui_modules import sidebar_inputs

# === Streamlit page configuration ===
st.set_page_config(
    page_title=f"{APP_NAME}",
    page_icon="🧪",
    layout="wide"
)


# === Header ===
st.title(f"🧪 {APP_NAME}")
st.caption(f"Version {APP_VERSION}")

log("Application started")

# === Load MDM (Attributes sheet) ===
with st.sidebar.expander("Data Model", icon="⚙️", expanded=False):
    if st.session_state.get('mdm_df') is None:
        st.session_state['mda_df'], st.session_state['facet_labels'] = load_mda()
        mdm_df = st.session_state['mda_df']
        if not mdm_df.empty:
            st.success("MDM loaded successfully ✅")
        else:
            st.error("Failed to load MDM ❌")

sidebar_inputs()


st.write("Welcome to the STF Generator application. Select a page on the left to get started.")
st.subheader("🔗 Navigation")
st.markdown(
    """
    Use the navigation menu to access:
    - **Input Viewer**: Upload and explore input data
    - **Form Generator**: Build test forms
    - **Exports**: Save results
    """
)
