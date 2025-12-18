
import os
import pickle
import shutil
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from fivedreg.data import prepare_data
from fivedreg.model import FiveDRegressor

# Constants
UPLOAD_DIR = Path("data_uploads")
MODEL_PATH = Path("model_latest.pt")
METADATA_PATH = Path("model_metadata.pkl")

# Global state
model_instance: Optional[FiveDRegressor] = None
model_metadata: dict = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load model if exists
    global model_instance, model_metadata
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    
    if MODEL_PATH.exists() and METADATA_PATH.exists():
        try:
            print(f"Loading existing model from {MODEL_PATH}...")
            # Load metadata to get hyperparameters/scaler
            with open(METADATA_PATH, "rb") as f:
                model_metadata = pickle.load(f)
            
            # Reconstruct model
            model_instance = FiveDRegressor(**model_metadata.get("hyperparameters", {}))
            
            # Load weights
            state_dict = torch.load(MODEL_PATH)
            model_instance.load_state_dict(state_dict)
            model_instance._is_fitted = True # Manually set fitted flag if needed, or rely on state
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Failed to load existing model: {e}")
            model_instance = None
            model_metadata = {}
    
    yield
    
    # Shutdown: Clean up or save logic if needed
    pass

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="5D Regression API", lifespan=lifespan)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", include_in_schema=False)
async def root():
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/docs")

class PredictRequest(BaseModel):
    features: List[List[float]]

class PredictResponse(BaseModel):
    predictions: List[float]

class TrainResponse(BaseModel):
    message: str
    metrics: dict

@app.get("/health")
async def health_check():
    return {
        "status": "active",
        "model_loaded": model_instance is not None
    }

@app.post("/upload")
async def upload_dataset(file: UploadFile = File(...)):
    if not file.filename.endswith(".pkl"):
        raise HTTPException(status_code=400, detail="Only .pkl files are allowed.")
    
    file_location = UPLOAD_DIR / "current_dataset.pkl"
    try:
        with open(file_location, "wb") as f:
            shutil.copyfileobj(file.file, f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to upload file: {e}")
        
    return {"message": f"Dataset uploaded successfully as {file_location.name}"}

class TrainRequest(BaseModel):
    hidden_layers: List[int] = [64, 32, 16]
    max_epochs: int = 200
    learning_rate: float = 1e-3

@app.post("/train")
async def train_model(request: TrainRequest):
    dataset_path = UPLOAD_DIR / "current_dataset.pkl"
    if not dataset_path.exists():
        raise HTTPException(status_code=400, detail="No dataset uploaded. Please upload a dataset first.")
        
    global model_instance, model_metadata
    
    try:
        # 1. Prepare Data
        prepared = prepare_data(dataset_path)
        
        # 2. Init Model
        hyperparameters = {
            "hidden_layers": tuple(request.hidden_layers),
            "max_epochs": request.max_epochs,
            "learning_rate": request.learning_rate,
            "verbose": True
        }
        model = FiveDRegressor(**hyperparameters)
        
        # 3. Train
        model.fit(
            prepared.X_train, prepared.y_train,
            X_val=prepared.X_val, y_val=prepared.y_val
        )
        
        # 4. Save Model & Metadata (including scaler!)
        torch.save(model.state_dict(), MODEL_PATH)
        
        model_metadata = {
            "hyperparameters": hyperparameters,
            "scaler": prepared.scaler,
            "imputer": prepared.imputer
        }
        with open(METADATA_PATH, "wb") as f:
            pickle.dump(model_metadata, f)
            
        model_instance = model
        
        # 5. Evaluate on Test Set
        test_mse = None
        if prepared.y_test is not None:
            test_preds = model.predict(prepared.X_test)
            test_mse = float(np.mean((prepared.y_test - test_preds) ** 2))
        
        return {
            "message": "Training completed successfully.",
            "metrics": {
                "train_samples": len(prepared.X_train),
                "val_samples": len(prepared.X_val),
                "test_mse": test_mse
            }
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Training failed: {e}")

@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    global model_instance, model_metadata
    if model_instance is None:
        raise HTTPException(status_code=400, detail="Model not trained or loaded.")
        
    try:
        input_data = np.array(request.features)
        
        # Validate input dimension
        if input_data.ndim != 2 or input_data.shape[1] != 5:
             raise HTTPException(status_code=400, detail=f"Input must be a list of 5D vectors. Got shape {input_data.shape}")

        # Preprocess
        scaler = model_metadata.get("scaler")
        imputer = model_metadata.get("imputer")
        
        if imputer:
            input_data = imputer.transform(input_data)
        if scaler:
            input_data = scaler.transform(input_data)
            
        # Predict
        preds = model_instance.predict(input_data)
        
        return {"predictions": preds.tolist()}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
