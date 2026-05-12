"""Default Streamlit Cloud entrypoint.

This file imports the app defined in main.py so deployments that expect
streamlit_app.py work out of the box.
"""

from main import *  # noqa: F401,F403
