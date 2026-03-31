from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SQLITE_DIR = DATA_DIR / "sqlite"
SQLITE_DB_PATH = SQLITE_DIR / "doc_sage.sqlite"
VECTOR_STORE_DIR = DATA_DIR / "vector_store"
TEMP_UPLOADS_DIR = DATA_DIR / "temp_uploads"


def ensure_app_directories() -> None:
    SQLITE_DIR.mkdir(parents=True, exist_ok=True)
    VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
    TEMP_UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
