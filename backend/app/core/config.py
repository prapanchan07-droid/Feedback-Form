import os
import yaml
try:
    from pydantic_settings import BaseSettings
except ImportError:
    from pydantic import BaseSettings
from typing import Dict, Any

class Settings(BaseSettings):
    PROJECT_NAME: str = "Bannari Amman Form Digitization & Verification Platform"
    API_V1_STR: str = "/api/v1"

    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./app.db")
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")

    STORAGE_TYPE: str = os.getenv("STORAGE_TYPE", "local")
    STORAGE_PATH: str = os.getenv("STORAGE_PATH", "./storage")

    CNN_MODEL_PATH: str = os.getenv("CNN_MODEL_PATH", "./ml/models/pretrained_resnet18.pth")
    CNN_CONFIDENCE_THRESHOLD: float = float(os.getenv("CNN_CONFIDENCE_THRESHOLD", "0.85"))
    CUTOFF_MIN: float = float(os.getenv("CUTOFF_MIN", "0.0"))
    CUTOFF_MAX: float = float(os.getenv("CUTOFF_MAX", "200.0"))
    MAX_UPLOAD_SIZE_MB: int = int(os.getenv("MAX_UPLOAD_SIZE_MB", "25"))

    FORM_TEMPLATE_PATH: str = os.getenv("FORM_TEMPLATE_PATH", "config/form_template.yaml")
    MODEL_CONFIG_PATH: str = os.getenv("MODEL_CONFIG_PATH", "config/model_config.yaml")

    class Config:
        case_sensitive = True
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()

def load_yaml_config(filepath: str) -> Dict[str, Any]:
    if not os.path.exists(filepath):
        return {}
    with open(filepath, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def get_form_template_config() -> Dict[str, Any]:
    return load_yaml_config(settings.FORM_TEMPLATE_PATH)

def get_model_config() -> Dict[str, Any]:
    return load_yaml_config(settings.MODEL_CONFIG_PATH)
