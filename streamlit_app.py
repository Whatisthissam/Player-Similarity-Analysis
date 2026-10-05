import os
import sys
import runpy

# Add Streamlit directory to system path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
STREAMLIT_DIR = os.path.join(ROOT_DIR, "Streamlit")
if STREAMLIT_DIR not in sys.path:
    sys.path.insert(0, STREAMLIT_DIR)

# Delegate to the main Streamlit application
APP_PATH = os.path.join(STREAMLIT_DIR, "app.py")
runpy.run_path(APP_PATH, run_name="__main__")
