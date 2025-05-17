# Логика ML-моделей


import os
import pickle
from typing import Dict, Any, Union
from dotenv import load_dotenv
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from pathlib import Path

# from .env
load_dotenv()
MODEL_DIR = os.getenv("MODEL_DIR", "./server/storage")
MAX_MODELS = int(os.getenv("MAX_MODELS", 10))

# supported models
MODEL_TYPES = {
    "logistic_regression": LogisticRegression,
    "random_forest": RandomForestClassifier,
    "gradient_boosting": GradientBoostingClassifier,
}

# Directory to save models
STORAGE_DIR = Path(MODEL_DIR)
STORAGE_DIR.mkdir(exist_ok=True)

# Cache for loaded models
_model_cache = {}


def fit(
    X: Union[np.ndarray, list], y: Union[np.ndarray, list], config: Dict[str, Any]
) -> Any:
    """
    Fit a machine learning model based on the provided configuration and save to disc under the specified name.;

    Parameters:
    - X: Features for training.
    - y: Target variable for training.
    - config: Configuration dictionary containing:
        - model_name: str, unique name for the model
        - model_type: str, type of model (e.g., 'logistic_regression', 'random_forest')
        - model_params: dict, hyperparameters for the model

    Returns:
    - model: Trained machine learning model.
    """
    model_name = config.get("model_name")
    model_type = config.get("model_type")
    model_params = config.get("model_params", {})

    if not model_name or not model_type:
        raise ValueError("Config must contain 'model_name' and 'model_type'")

    if model_type not in MODEL_TYPES:
        raise ValueError(
            f"Model type '{model_type}' is not supported. Supported types are: {', '.join(MODEL_TYPES.keys())}"
        )

    # Check limit of models
    existing_models = len(list(STORAGE_DIR.glob("*.pkl")))
    model_path = STORAGE_DIR / f"{model_name}.pkl"
    if not model_path.exists() and existing_models >= MAX_MODELS:
        raise ValueError(
            f"Maximum number of models ({MAX_MODELS}) reached. Please remove some models before adding new ones."
        )

    # Model initialization
    model_class = MODEL_TYPES[model_type]
    model = model_class(**model_params)

    # Model training
    model.fit(X, y)

    # Save the model to disk
    model_path = STORAGE_DIR / f"{model_name}.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    # Cache the model
    _model_cache[model_name] = model

    return model


def predict(y: Union[np.ndarray, list], config: Dict[str, Any]) -> np.ndarray:
    """
    Predict using the trained and downloaded model by its name.

    Args:
        y: Target variable for prediction.
        config: Configuration dictionary containing model parameters.
            - model_name: str, unique name for the model

    Returns:
        - predictions: Model predictions
    """
    model_name = config.get("model_name")

    if not model_name:
        raise ValueError("Config must contain 'model_name'")
    if model_name not in _model_cache:
        try:
            load(config)
        except FileNotFoundError:
            raise FileNotFoundError(f"Model '{model_name}' not found on disk")

    model = _model_cache.get(model_name)
    if model is None:
        raise ValueError(f"Model '{model_name}' not found in cache.")

    return model.predict(y)


def load(config: Dict[str, Any]) -> None:
    """
    Download the trained model by it name for inference.

    Args:
        config: Configuration dictionary containing model parameters.
            - model_name: str, unique name for the model
    """
    model_name = config.get("model_name")

    if not model_name:
        raise ValueError("Config must contain 'model_name'")

    # Check if the model is already loaded
    if model_name in _model_cache:
        return _model_cache[model_name]

    # Load the model from disk
    model_path = STORAGE_DIR / f"{model_name}.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Model '{model_name}' not found on disk.")

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # Cache the loaded model
    _model_cache[model_name] = model


def unload(config: Dict[str, Any]) -> None:
    """
    Unload the loaded model by its name.

    Args:
        config: Configuration dictionary containing model parameters.
            - model_name: str, unique name for the model
    """
    model_name = config.get("model_name")
    if not model_name:
        raise ValueError("Config must contain 'model_name'")

    # Unload the model from cache
    _model_cache.pop(model_name, None)


def remove(config: Dict[str, Any]) -> None:
    """
    Remove the trained model from disc by its name.

    Args:
        config: Configuration dictionary containing model parameters.
    """
    model_name = config.get("model_name")
    if not model_name:
        raise ValueError("Config must contain 'model_name'")

    # Unload the model from cache
    _model_cache.pop(model_name, None)
    # Remove the model from disk
    model_path = STORAGE_DIR / f"{model_name}.pkl"
    if model_path.exists():
        model_path.unlink()


def remove_all() -> None:
    """
    Remove all trained models from disc.
    """
    # Clear the cache
    _model_cache.clear()

    # Remove all models from disk
    for model_path in STORAGE_DIR.glob("*.pkl"):
        model_path.unlink()
