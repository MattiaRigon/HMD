import os
from pathlib import Path
from sqlalchemy import create_engine

# API URL
API_URL = "https://www.themealdb.com/api/json/v1/1/search.php?s="

# Database configuration
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = Path(f"{CURRENT_DIR}/meals.db")  # Save database in a subdirectory called 'data'
DB_PATH.parent.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists
engine = create_engine(f"sqlite:///{DB_PATH}")
