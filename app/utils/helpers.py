 #app/utils/helpers.py

"""
Helper utilities for logging, formatting, etc.
"""

import logging
import datetime
import os

LOG_FILE = "log.txt"

# Setup logger
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def log_event(msg):
    """Log custom event to file"""
    logging.info(msg)

def timestamp():
    """Returns current time string"""
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def ensure_dir(path):
    """Ensure folder exists"""
    if not os.path.exists(path):
        os.makedirs(path)
