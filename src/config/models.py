# src/config/models.py

"""
Centralized model registry for the sports analytics system.

Provides a single source of truth for all model paths used across the pipeline.
All paths are computed dynamically relative to the project root.
"""

from pathlib import Path
from typing import Dict


def _get_project_root() -> Path:
    """
    Compute the project root directory dynamically.

    Assumes this file is located at: <project_root>/src/config/models.py

    Returns:
        Path to project root directory
    """
    # This file is at src/config/models.py
    # Go up two levels: config -> src -> project_root
    return Path(__file__).parent.parent.parent


# Model registry: maps model names to relative paths from project root
_MODEL_REGISTRY: Dict[str, str] = {
    # Fine-tuned YOLOv8 model for football + player detection
    "ball": "models/finetuned/yolov8n_v3.pt",

    # Base pretrained YOLOv8 model (for experiments or future fine-tuning)
    "pretrained": "models/pretrained/yolov8n.pt",
}


def get_model(name: str) -> Path:
    """
    Get the absolute path to a registered model.

    Args:
        name: Model identifier (e.g., "ball", "pretrained")

    Returns:
        Absolute Path to the model file

    Raises:
        ValueError: If model name is not registered

    Examples:
        >>> ball_model = get_model("ball")
        >>> pretrained_model = get_model("pretrained")
    """
    if name not in _MODEL_REGISTRY:
        available = ", ".join(f'"{k}"' for k in _MODEL_REGISTRY.keys())
        raise ValueError(
            f'Unknown model name: "{name}". '
            f"Available models: {available}"
        )

    project_root = _get_project_root()
    model_relative_path = _MODEL_REGISTRY[name]

    return project_root / model_relative_path
