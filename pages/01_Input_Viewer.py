# STFGenerator/pages/01_Input_Viewer.py

import streamlit as st


from config.settings import log, MDM_FILE
from modules.utils import merge_df, show_n_select_ecs
from modules.ui_modules import sidebar_inputs

# from modules.loader import load_excel

# Page configuration
st.set_page_config(page_title="Input Viewer", page_icon="👀", layout="wide")

# --- Page start ---
# log("Opening page: 01_Input_Viewer", level="INFO")

st.title("👀 Input Viewer")

sidebar_inputs()

if st.session_state.inputs_integrated:
    try:
        show_n_select_ecs(st.session_state.dataframes)

    except Exception as e:
        st.error(f"Error merging data: {e}")
        log(f"Exception during merge: {e}", level="ERROR")
