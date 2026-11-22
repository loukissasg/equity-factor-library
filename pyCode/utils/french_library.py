# ABOUTME: Downloads a zipped CSV file from Ken French's data library.
"""
French publishes each dataset as a zip containing one CSV with a text header and
several tables one after the other. This helper returns the CSV as a list of lines;
each download script parses the table it needs.
"""

import io
import zipfile

import requests

FRENCH_FTP = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"


def download_french_csv(zip_name):
    """Download FRENCH_FTP + zip_name and return the CSV inside as a list of lines."""
    response = requests.get(FRENCH_FTP + zip_name, timeout=60)
    response.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        text = z.read(z.namelist()[0]).decode("utf-8", errors="ignore")
    return text.splitlines()
