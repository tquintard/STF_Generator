# STFGenerator/pages/01_Input_Viewer.py

import streamlit as st


from config.settings import log, MDM_FILE
from modules.utils import show_n_select_ecs


# from modules.loader import load_excel

# Page configuration
st.set_page_config(page_title="HPC SG Portal", page_icon="⛩️", layout="wide")

# --- Page start ---
# log("Opening page: 01_Input_Viewer", level="INFO")

st.title("⛩️ HPC SG Portal")

if "inputs_integrated" not in st.session_state:
    st.session_state.inputs_integrated = False
if st.session_state.inputs_integrated:
    show_n_select_ecs(st.session_state.dataframes)
