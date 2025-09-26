# STFGenerator/config/settings.py

import os


# === BASE DIRECTORIES ===
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# === PATHS ===
INPUT_DIR = os.path.join(BASE_DIR, "inputs")
MDM_DIR = os.path.join(BASE_DIR, "mdm")
MDM_FILE = os.path.join(MDM_DIR, "STFG_MDA.xlsx")
MODULES_DIR = os.path.join(BASE_DIR, "modules")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
LOG_FILE = os.path.join(BASE_DIR, "stfg_app.log")

# === APP SETTINGS ===
APP_NAME = "STF Generator"
APP_VERSION = "0.1"

# === LOGGER ===
def log(message: str, level: str = "INFO"):
    """
    Simple logger that writes to both console and log file.

    Args:
        message (str): message to log
        level (str): log level ("INFO", "WARNING", "ERROR", "DEBUG")
    """
    import datetime

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted = f"[{timestamp}] [{level}] {message}"

    # Console
    print(formatted)

    # Log file
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(formatted + "\n")


