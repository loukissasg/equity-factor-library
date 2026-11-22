# ABOUTME: Opens a WRDS connection using the username in .env and the password in pgpass.
"""
The password is never in the code: it lives in the pgpass file created the first time
you ran `python -c "import wrds; db = wrds.Connection()"`. The username comes from .env
(see dotenv.template).
"""

import os

import wrds
from dotenv import load_dotenv

from config import ENV_FILE


def get_wrds_connection():
    """Return an open wrds.Connection. Close it with .close() when done."""
    load_dotenv(ENV_FILE)
    username = os.getenv("WRDS_USERNAME")
    if not username:
        raise SystemExit("WRDS_USERNAME not found. Copy dotenv.template to .env and fill it in.")
    return wrds.Connection(wrds_username=username)
