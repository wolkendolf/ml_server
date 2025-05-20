# FastAPI сервер
import os
import multiprocessing
from typing import Dict, Any, Union
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from dotenv import load_dotenv
import numpy as np
from server.models import fit, predict, load, unload, remove, remove_all

# Download .env file
load_dotenv()
CPU_CORES = int(os.getenv("CPU_CORES", 4))  # 4 cores by default
MAX_PROCESSES = max(1, CPU_CORES - 1)
MAX_REQUESTS = CPU_CORES * 2  # limit of concurrent requests for inference
MODEL_DIR = os.getenv("MODEL_DIR", "./server/storage")
MAX_MODELS = int(os.getenv("MAX_MODELS", 10))

app = FastAPI(title="ML Model Server", version="1.0.0")

active_processes = multiprocessing.Value("i", 0)
active_requests = multiprocessing.Value("i", 0)


class TrainRequest(BaseModel):
    X: list
    y: list
    config: Dict[str, Any]


class PredictRequest(BaseModel):
    y: list
    config: Dict[str, Any]


class ConfigRequest(BaseModel):
    config: Dict[str, Any]


def run_fit(
    X: Union[np.ndarray, list], y: Union[np.ndarray, list], config: Dict[str, Any]
):
    """
    Start a background process to fit a model.
    This function is called when a new model is being trained.
    It increments the active_processes counter to track the number of concurrent training processes.
    """
    try:
        fit(X, y, config)
    finally:
        with active_processes.get_lock():
            active_processes.value -= 1
            print("run_fit_func: ", active_processes.value)


@app.post("/fit")
async def fit_model(request: TrainRequest, background_tasks: BackgroundTasks):
    """
    Train a model in a separate process.
    """
    try:
        with active_processes.get_lock():
            if active_processes.value >= MAX_PROCESSES:
                raise HTTPException(
                    status_code=429,
                    detail=f"No available cores for training. Max processes: {MAX_PROCESSES}",
                )

        process = multiprocessing.Process(
            target=run_fit,
            args=(request.X, request.y, request.config),
        )
        process.start()
        background_tasks.add_task(process.join)

        active_processes.value += 1
        print("fit_model: ", active_processes.value)

        return {
            "message": f"Training started for model_name '{request.config.get('model_name')}'"
        }
    except Exception as e:
        with active_processes.get_lock():
            active_processes.value -= 1
            print("fit_model_except: ", active_processes.value)
        if "No available cores" in str(e):
            raise HTTPException(status_code=429, detail=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict")
async def predict_model(request: PredictRequest):
    """
    Predict using a trained model.
    """
    with active_requests.get_lock():
        if active_requests.value >= MAX_REQUESTS:
            raise HTTPException(
                status_code=429,
                detail=f"No available cores for inference. Max requests: {MAX_REQUESTS}",
            )
        active_requests.value += 1
        print("predict_model: ", active_processes.value)

    try:
        predictions = predict(request.y, request.config)
        return {"predictions": predictions.tolist()}
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=f"Model not found: {str(e)}")
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error during prediction: {str(e)}"
        )
    finally:
        with active_requests.get_lock():
            active_requests.value -= 1
            print("predict_model_finally: ", active_processes.value)


@app.post("/load")
async def load_model(request: ConfigRequest):
    """
    Load a model for inference.
    """
    try:
        load(request.config)
        return {
            "message": f"Model '{request.config.get('model_name')}' loaded successfully."
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/unload")
async def unload_model(request: ConfigRequest):
    """
    Unload a trained model from memory.
    """
    try:
        unload(request.config)
        return {
            "message": f"Model '{request.config.get('model_name')}' unloaded successfully."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/remove")
async def remove_model(request: ConfigRequest):
    """
    Remove a trained model from disk.
    """
    try:
        remove(request.config)
        return {
            "message": f"Model '{request.config.get('model_name')}' removed successfully."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/remove_all")
async def remove_all_models():
    """
    Remove all trained models from disk.
    """
    try:
        remove_all()
        return {"message": "All models removed successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/status")
async def get_status():
    """
    Get the status of the server.
    """
    return {
        "cpu_cores": CPU_CORES,
        "max_processes": MAX_PROCESSES,
        "active_processes": active_processes.value,
        "active_requests": active_requests.value,
        "max_requests": MAX_REQUESTS,
        "model_dir": MODEL_DIR,
        "max_models": MAX_MODELS,
    }
