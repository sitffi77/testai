"""
Configuration module for Pain Diagnosis Application.

Contains paths, hyperparameters, database settings, and other constants.
Sensitive data (like DB_PASSWORD) should be provided via environment variables.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file if exists
load_dotenv()

# Base directory
BASE_DIR = Path(__file__).parent.absolute()

# Model paths
MODEL_DIR = BASE_DIR / "models" / "saved"
DEFAULT_MODEL_PATH = MODEL_DIR / "pain_classifier.joblib"

# Database configuration
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", "3306"))
DB_NAME = os.getenv("DB_NAME", "pain_diagnosis")
DB_USER = os.getenv("DB_USER", "root")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")

DATABASE_URL = f"mysql+mysqlconnector://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# ML hyperparameters
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Feature configuration
NUMERICAL_FEATURES = [
    "age",
    "pain_intensity",
    "duration_days",
    "frequency_per_week",
    "sleep_hours",
    "stress_level",
]

CATEGORICAL_FEATURES = [
    "gender",
    "pain_location",
    "pain_type",
    "trigger_factor",
    "relief_factor",
    "medication_use",
]

ALL_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES

# Pain classes for classification
PAIN_CLASSES = [
    "nociceptive",
    "neuropathic",
    "nociplastic",
    "mixed",
]

# UI settings
WINDOW_TITLE = "Система поддержки диагностирования болевых синдромов"
WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 800

# Logging configuration
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_FILE = BASE_DIR / "app.log"

# SHAP visualization settings
SHAP_FIGURE_DPI = 100
SHAP_FIGURE_SIZE = (8, 6)
