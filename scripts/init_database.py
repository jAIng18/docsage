import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.database import initialize_database
from src.paths import SQLITE_DB_PATH


if __name__ == "__main__":
    initialize_database()
    print(f"Tables created successfully at {SQLITE_DB_PATH}.")
