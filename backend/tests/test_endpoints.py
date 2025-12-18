import pickle
import pytest
import numpy as np
from fastapi.testclient import TestClient
from unittest.mock import patch
from pathlib import Path

# Import the app. 
# Note: We need to make sure we don't accidentally import main in a way that runs global code if it had any side effects, 
# but here it's safe.
from main import app

# --- Fixtures ---

@pytest.fixture
def client():
    return TestClient(app)

@pytest.fixture
def mock_fs(tmp_path):
    """
    Patches the global path constants in main.py to use a temporary directory.
    This ensures tests don't overwrite real data/models.
    """
    # Create subdirs
    upload_dir = tmp_path / "data_uploads"
    upload_dir.mkdir()
    
    model_path = tmp_path / "model_latest.pt"
    meta_path = tmp_path / "model_metadata.pkl"

    # Patch the constants using string paths to where they are used
    with patch("main.UPLOAD_DIR", upload_dir), \
         patch("main.MODEL_PATH", model_path), \
         patch("main.METADATA_PATH", meta_path):
        yield {
            "upload_dir": upload_dir,
            "model_path": model_path,
            "meta_path": meta_path
        }

@pytest.fixture
def dummy_pkl_content():
    """Returns bytes of a valid pickle dataset."""
    rng = np.random.default_rng(42)
    X = rng.normal(size=(20, 5)).astype(np.float32)
    y = rng.normal(size=(20,)).astype(np.float32)
    data = {"X": X, "y": y}
    return pickle.dumps(data)

# --- Tests ---

def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "active"

def test_upload_flow(client, mock_fs, dummy_pkl_content):
    # 1. Upload
    files = {"file": ("dataset.pkl", dummy_pkl_content, "application/octet-stream")}
    response = client.post("/upload", files=files)
    assert response.status_code == 200
    assert "successfully" in response.json()["message"]
    
    # Verify file exists
    assert (mock_fs["upload_dir"] / "current_dataset.pkl").exists()

def test_train_flow(client, mock_fs, dummy_pkl_content):
    # 1. Upload first (training requires existing dataset)
    files = {"file": ("dataset.pkl", dummy_pkl_content, "application/octet-stream")}
    client.post("/upload", files=files)
    
    # 2. Train
    payload = {
        "hidden_layers": [16, 8],
        "max_epochs": 2,
        "learning_rate": 0.01
    }
    response = client.post("/train", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "metrics" in data
    assert data["metrics"]["train_samples"] > 0
    
    # Verify model artifact creation
    assert mock_fs["model_path"].exists()
    assert mock_fs["meta_path"].exists()

def test_predict_flow(client, mock_fs, dummy_pkl_content):
    # 1. Upload & Train to ensure model is loaded in memory
    files = {"file": ("dataset.pkl", dummy_pkl_content, "application/octet-stream")}
    client.post("/upload", files=files)
    client.post("/train", json={"max_epochs": 1})
    
    # 2. Predict
    # Create input: list of lists
    input_features = [[0.1, 0.2, 0.3, 0.4, 0.5]]
    response = client.post("/predict", json={"features": input_features})
    
    assert response.status_code == 200
    preds = response.json()["predictions"]
    assert len(preds) == 1
    assert isinstance(preds[0], float)

def test_predict_without_model(client, mock_fs):
    # Ensure no model is loaded by clearing global state if necessary, 
    # but fresh client + fresh mock_fs implies clean slate usually?
    # Actually, `main.model_instance` is global. TestClient reuses the app instance.
    # We might need to manually reset the global variable in main.py if we want true isolation,
    # or rely on the fact that `lifespan` runs per app instance? 
    # FastAPI `TestClient` DOES run lifespan events.
    # However, global variables in `main` persist across tests in the same process 
    # unless we explicitly reset them.
    
    # Let's force reset for safety
    with patch("main.model_instance", None):
        response = client.post("/predict", json={"features": [[0]*5]})
        assert response.status_code == 400
        assert "Model not trained" in response.json()["detail"]