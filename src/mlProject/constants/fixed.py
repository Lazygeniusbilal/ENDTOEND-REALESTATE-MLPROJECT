from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parent.parent.parent
CONFIG_FILE_PATH = BASE_DIR / "config/config.yaml"
PARAMS_FILE_PATH = BASE_DIR / "params.yaml"
SCHEMA_FILE_PATH = BASE_DIR / "schema.yaml"