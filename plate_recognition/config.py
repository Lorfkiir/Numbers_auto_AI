"""Настройки путей проекта.

Пути можно переопределить переменными окружения:
OCR_TRAIN_DIR, OCR_VALID_DIR, OCR_TEST_DIR,
REC_TRAIN_DIR, REC_VALID_DIR, REC_TEST_DIR, MODEL_DIR.
"""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = Path(os.getenv("MODEL_DIR", PROJECT_ROOT / "models" / "checkpoints"))

PATH_TO_TRAIN_DATASET = Path(os.getenv("OCR_TRAIN_DIR", DATA_DIR / "OCR_data" / "train"))
PATH_TO_VALID_DATASET = Path(os.getenv("OCR_VALID_DIR", DATA_DIR / "OCR_data" / "val"))
PATH_TO_TEST_DATASET = Path(os.getenv("OCR_TEST_DIR", DATA_DIR / "OCR_data" / "test"))

PATH_TO_TRAIN_DATASET1 = Path(os.getenv("REC_TRAIN_DIR", DATA_DIR / "REC_data" / "train"))
PATH_TO_VALID_DATASET1 = Path(os.getenv("REC_VALID_DIR", DATA_DIR / "REC_data" / "val"))
PATH_TO_TEST_DATASET1 = Path(os.getenv("REC_TEST_DIR", DATA_DIR / "REC_data" / "test"))

PATH_TO_SAVE_MODEL = str(MODEL_DIR)
for directory in (PATH_TO_SAVE_MODEL,):
    Path(directory).mkdir(parents=True, exist_ok=True)
