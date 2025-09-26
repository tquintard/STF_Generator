# STFGenerator/pages/01_Input_Viewer.py

import streamlit as st


from config.settings import log, MDM_FILE
from modules.utils import merge_df, show_n_select_ecs
from modules.ui_modules import upload_sidebar

#from modules.loader import load_excel

# Page configuration
st.set_page_config(page_title="Input Viewer", page_icon="👀",layout="wide")

# --- Page start ---
#log("Opening page: 01_Input_Viewer", level="INFO")

st.title("👀 Input Viewer")


uploaded_files = upload_sidebar()

if uploaded_files:
    try:
        show_n_select_ecs(uploaded_files)
        
    except Exception as e:
        st.error(f"Error merging data: {e}")
        log(f"Exception during merge: {e}", level="ERROR")
