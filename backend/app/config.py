import os
from pathlib import Path
from dotenv import load_dotenv

dotenv_path = Path(os.getenv("DOTENV_PATH", ".env.dev"))
load_dotenv(dotenv_path=dotenv_path)

def get_env_var_or_raise(key: str) -> str:
    value = os.getenv(key)
    if not value:
        raise ValueError(f"Missing environment variable {key}")
    return value

DB_URL = get_env_var_or_raise("DATABASE_URL")