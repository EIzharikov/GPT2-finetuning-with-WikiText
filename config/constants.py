"""
Useful constant variables
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

PREDICTION_PATH = PROJECT_ROOT / 'preds'

BASIC_MODEL_PRED_PATH = PREDICTION_PATH / 'basic_model.csv'
LORA_MODEL_PRED_PATH = PREDICTION_PATH / 'lora_model.csv'

BASIC_MODEL_PATH = PROJECT_ROOT / 'basic_model'
LORA_MODEL_PATH = PROJECT_ROOT / 'lora_model'
