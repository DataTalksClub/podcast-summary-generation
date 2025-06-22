import subprocess
import sys
import os

# Get the full path to your Streamlit app
script_path = os.path.join(os.path.dirname(__file__), "app.py")

# Run Streamlit using the current Python executable
subprocess.run([sys.executable, "-m", "streamlit", "run", script_path])