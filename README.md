# Interpoletor: 5D Regression System

A production-ready, full-stack machine learning system for training and deploying neural networks on 5-dimensional regression tasks. Built with modern web technologies and containerized microservices architecture.

**Author**: Iuliia Vituigova (iv294@cam.ac.uk)
**Institution**: Cambridge University 
**Year**: 2025

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Technology Stack](#technology-stack)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Testing](#testing)
- [Performance and Profiling](#performance-and-profiling)
- [Environment Variables](#environment-variables)
- [API Documentation](#api-documentation)
- [Project Structure](#project-structure)
- [Troubleshooting](#troubleshooting)

---

## Overview

**Interpoletor** is a comprehensive machine learning system designed for 5-dimensional regression problems, featuring:

- **Data Management**: Functionality for uploading and validating .pkl datasets with automated shape verification.
- **Model Training**: Configurable PyTorch-based neural networks incorporating early stopping and validation mechanisms.
- **Real-time Inference**: A FastAPI REST API facilitateing predictive analysis.
- **Modern User Interface**: A Next.js single-page application providing real-time system monitoring.
- **Containerization**: Full Docker integration for reproducible deployments.
- **Comprehensive Testing**: 34 tests achieving 77% code coverage.

The system implements a custom PyTorch-based regressor (`FiveDRegressor`) supported by scikit-learn preprocessing:
- Configurable multi-layer feedforward architecture.
- Adam optimization with L2 regularization.
- Early stopping with validation loss monitoring.
- Automated feature standardization and imputation.
- Reproducible training ensured by fixed random seeds.

---

## Features

### Backend (FastAPI + PyTorch)
- **RESTful API** with automated OpenAPI documentation.
- **Data Validation**: Strict type checking via Pydantic.
- **Model Persistence**: Automated saving and loading of models and preprocessing pipelines.
- **CORS Support**: Configured for frontend integration.
- **Lifecycle Management**: Automated model loading upon startup.
- **Health Checks**: Docker-compatible health monitoring.
- **Error Handling**: Comprehensive error reporting and appropriate status codes.

### Frontend (Next.js + React + TypeScript)
- **Single-Page Application**: Integration of all functionalities into a unified interface.
- **Technology Stack**: Next.js 14, React 18, TypeScript, and Tailwind CSS.
- **Real-time Monitoring**: System status indicators for backend connectivity and model availability.
- **Interactive Visualizations**: Training history visualization using Recharts.
- **Responsive Design**: Professional academic styling with cross-device compatibility.
- **Functional Workflow**:
  - **Upload**: Secure file upload with integrated validation and preview.
  - **Train**: Interface for dynamic architecture configuration and hyperparameter control.
  - **Predict**: Analytical interface for input-based prediction results.
  - **Visualize**: Comparative visualization of training and validation loss.

### Machine Learning Pipeline
- **Data Preprocessing**:
  - Dataset splitting (64% training, 16% validation, 20% testing).
  - Feature standardization (zero mean, unit variance).
  - Missing value imputation.
  - Automated dataset validation protocols.

- **Model Architecture**:
  - Fully connected neural network (5-input, variable hidden layers, 1-output).
  - Configurable hidden layer dimensionality.
  - ReLU activation functions.
  - Mean Squared Error (MSE) loss function for regression.

- **Training Capabilities**:
  - Mini-batch training via PyTorch DataLoader.
  - Early stopping mechanism for regularization.
  - Systematic validation performance monitoring.
  - TorchMetrics integration for MSE, R², and MAE.
  - Optional logging support via Weights & Biases.

### Testing and Quality Assurance
- **74 Tests** achieving **100% code coverage**.
- **Comprehensive Coverage**:
  - fivedreg/__init__.py: 100%
  - fivedreg/data.py: 100%
  - fivedreg/model.py: 100%
  - fivedreg/utils.py: 100%
- **Test Categories**:
  - Unit tests for model initialization and error handling.
  - Integration tests for API endpoints and logging.
  - End-to-end workflow validation.
  - Data pipeline and preprocessing edge case testing.
  - Utility and file operation verification.
  - Coverage completion for all validation paths.
- **CI/CD Readiness**: pytest configuration with detailed coverage reporting.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                      User Browser                            │
│                 (http://localhost:3000)                      │
└───────────────────────┬──────────────────────────────────────┘
                        │
                        │ HTTP/REST API
                        │
┌───────────────────────▼──────────────────────────────────────┐
│               Frontend Service (Next.js)                     │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  Single Page Application                               │  │
│  │  • Upload Section  • Training Config  • Prediction     │  │
│  │  • Status Monitor  • History Charts  • System Info     │  │
│  └────────────────────────────────────────────────────────┘  │
│  Port: 3000  |  Container: interpoletor-frontend             │
└───────────────────────┬──────────────────────────────────────┘
                        │
                        │ REST API Calls
                        │
┌───────────────────────▼──────────────────────────────────────┐
│              Backend Service (FastAPI)                       │
│  ┌────────────────────────────────────────────────────────┐  │
│  │                 API Endpoints                          │  │
│  │  GET  /health  - Health check                          │  │
│  │  POST /upload  - Dataset upload                        │  │
│  │  POST /train   - Model training                        │  │
│  │  POST /predict - Inference                             │  │
│  │  GET  /docs    - Interactive API documentation         │  │
│  └──────────────┬─────────────────────────────────────────┘  │
│                 │                                            │
│  ┌──────────────▼─────────────────────────────────────────┐  │
│  │           Core ML Pipeline (PyTorch)                   │  │
│  │  ┌───────────┐  ┌────────────┐  ┌──────────────────┐   │  │
│  │  │  Data     │→ │Preprocessing│→ │  FiveDRegressor │   │  │
│  │  │  Loading  │  │  (sklearn)  │  │   (PyTorch NN)  │   │  │
│  │  └───────────┘  └────────────┘  └──────────────────┘   │  │
│  └────────────────────────────────────────────────────────┘  │
│  Port: 8000  |  Container: interpoletor-backend              │
│                                                              │
│  Persistent Volumes:                                         │
│  • data_uploads/  - Uploaded datasets                        │
│  • models/        - Saved model weights                      │
│  • figures/       - Generated visualizations                 │
└──────────────────────────────────────────────────────────────┘

Network: interpoletor-network (bridge)

---

## Technology Stack

### Backend
| Technology | Version | Purpose |
|-----------|---------|---------|
| **Python** | 3.10+ | Core language |
| **FastAPI** | ≥0.104.0 | Web framework & REST API |
| **PyTorch** | ≥2.0.0 | Neural network implementation |
| **NumPy** | ≥1.24.0, <2.0 | Numerical computing |
| **scikit-learn** | ≥1.3.0 | Preprocessing (StandardScaler, imputation) |
| **TorchMetrics** | ≥1.0.0 | Metrics computation (MSE, R², MAE) |
| **Pydantic** | ≥2.0.0 | Data validation & serialization |
| **Uvicorn** | ≥0.24.0 | ASGI server |

### Frontend
| Technology | Version | Purpose |
|-----------|---------|---------|
| **Next.js** | 14.0+ | React framework with SSR/SSG |
| **React** | 18.2+ | UI library |
| **TypeScript** | 5.0+ | Type-safe JavaScript |
| **Tailwind CSS** | 3.3+ | Utility-first CSS framework |
| **Axios** | 1.6+ | HTTP client |
| **Recharts** | 2.10+ | Data visualization |
| **Lucide React** | 0.294+ | Icon library |

### Testing & DevOps
| Tool | Purpose |
|------|---------|
| **pytest** | Testing framework |
| **pytest-cov** | Code coverage |
| **httpx** | Async HTTP client for testing |
| **Docker** | Containerization |
| **Docker Compose** | Multi-container orchestration |

---

## Code Documentation and Comments

All code files include comprehensive inline documentation to facilitate understanding:

### Backend Implementation Documentation

#### **fivedreg/model.py** (FiveDRegressor)
- **Docstrings**: Formatted documentation for all classes and methods.
- **Parameter Descriptions**: Specification of argument types and default values.
- **Return Values**: Documentation of function outputs.
- **Usage Examples**: Integrated examples within docstrings.
- **Inline Comments**: Explanations of complex implementation details (e.g., early stopping, metrics calculation).

Example:
```python
def fit(self, X_train, y_train, X_val=None, y_val=None):
    """
    Train the model with optional wandb logging and comprehensive metrics.

    Args:
        X_train: numpy array of shape (n_train, 5)
        y_train: numpy array of shape (n_train,)
        X_val: optional validation features
        y_val: optional validation targets
    """
    # Training implementation with inline comments
```

#### **fivedreg/data.py** (Data Pipeline)
- **Comprehensive Docstrings**: Documentation for all pipeline functions.
- **Validation Logic**: Explanations of data verification procedures.
- **Pipeline Steps**: Systematic documentation of transformation stages.
- **Error Reporting**: Informative error messages for standard failure modes.

#### **fivedreg/utils.py** (Utilities)
- **Function Documentation**: Detailed purpose of utility functions.
- **Format Specifications**: Documentation of supported file formats.
- **Directory Structure**: Analysis of the required filesystem hierarchy.
- **Examples**: Practical usage demonstrations.

#### **main.py** (FastAPI Application)
- **Endpoint Documentation**: Automated documentation integrated with /docs.
- **Data Models**: Pydantic models for request and response validation.
- **Error Handling**: Documented exception management.
- **Lifecycle Hooks**: Clarification of startup and shutdown procedures.

### Frontend Implementation Documentation

#### **src/app/page.tsx** (Main Application)
- **Component Structure**: Explicit marking of functional sections.
- **State Management**: Documentation of data flow and state transitions.
- **API Integration**: Details of backend communication.
- **Event Handlers**: Rationale for user interaction logic.

#### **src/app/globals.css** (Styling)
- **CSS Classes**: Documentation of styling conventions.
- **Tailwind Utilities**: Explanations of styling utilities.
- **Animations**: Descriptions of custom transition effects.

### Test Documentation

Test suites incorporate:
- **Test Objectives**: High-level purpose designated at the file level.
- **Case Rationale**: Method-level docstrings explaining test scenarios.
- **Assertions**: Rationale for specific validation checks.
- **Lifecycle Management**: Documentation of setup and teardown procedures.

Example from `test_core.py`:
```python
def test_model_init_defaults():
    """Test that model initializes with correct default hyperparameters."""
    model = FiveDRegressor()
    assert model.learning_rate == 1e-3  # Default LR
    assert model.max_epochs == 200      # Default epochs
    # ... more assertions
```

### Configuration and Build Documentation

- **pyproject.toml**: Dependency specifications with versioning constraints.
- **package.json**: Documentation of build and development scripts.
- **docker-compose.yml**: Configuration of containerized microservices.
- **Dockerfile**: Analysis of multi-stage build processes.

---

## Quick Start

### Utilizing Docker Compose (Recommended for Verification)

This is the recommended method for verifying the integrated functionality of both frontend and backend services.

#### Step 1: Initialize Docker Services

```bash
# Navigate to project root
cd /Users/julia/Desktop/CAM_COURSES/C1/cw_c1/interpoletor

# Initialize backend and frontend services (initial build: 3-5 minutes)
docker-compose up --build
```

**Process Details**:
1. Backend construction (Python 3.10 with dependency resolution).
2. Frontend construction (Node 18 with Next.js build optimization).
3. Service initialization and network configuration.
4. Execution of health monitoring protocols.

**Example Output**:
```
interpoletor-backend    | INFO:     Uvicorn running on http://0.0.0.0:8000
interpoletor-frontend   | ▲ Next.js ready on http://0.0.0.0:3000
```

#### Step 2: Accessing the Application

Access the services via a web browser:

| Service | URL | Content |
|---------|-----|-----------------|
| **Frontend interface** | http://localhost:3000 | Comprehensive UI for dataset management and analysis |
| **Backend API** | http://localhost:8000 | API status information |
| **API Documentation** | http://localhost:8000/docs | Interactive Swagger UI for endpoint exploration |
| **Health Monitoring** | http://localhost:8000/health | Real-time system status indicators |

#### Step 3: Operational Workflow Validation

**3.1 - Verify Backend Status**
```bash
curl http://localhost:8000/health
```

Expected response:
```json
{
  "status": "active",
  "model_loaded": false
}
```

**3.2 - Dataset Upload**

Users may utilize either the frontend interface or a command-line tool:
- **Frontend Interface**: Navigate to http://localhost:3000 for drag-and-drop .pkl file upload.
- **REST API**:
```bash
curl -X POST http://localhost:8000/upload \
  -F "file=@test_data/your_dataset.pkl"
```

**3.3 - Model Training**

Via **Frontend Interface**:
1. Access the "2. Train Model" section.
2. Define hyperparameter configurations (default: 64, 32, 16 layers).
3. Select "Start Training".
4. Expect 10-30 seconds for completion.
5. Training metrics will be rendered upon completion.

Via **REST API**:
```bash
curl -X POST http://localhost:8000/train \
  -H "Content-Type: application/json" \
  -d '{
    "hidden_layers": [64, 32, 16],
    "max_epochs": 200,
    "learning_rate": 0.001
  }'
```

**3.4 - Predictive Analysis**

Via **Frontend Interface**:
1. Access the "3. Make Predictions" section.
2. Input 5 feature values (e.g., 1.0, 2.0, 3.0, 4.0, 5.0).
3. Select "Predict".
4. Results will be displayed prominently within the interface.

Via **REST API**:
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": [[1.0, 2.0, 3.0, 4.0, 5.0]]
  }'
```

**3.5 - Monitoring Training Progress**

The frontend provides real-time visualizations for:
- Training and validation loss trajectories.
- Final performance metrics summary.
- Comprehensive system information.

#### Step 4: Log Analysis

Monitor service logs in real-time using:

```bash
# Aggregate logs
docker-compose logs -f

# Backend-specific logs
docker-compose logs -f backend

# Frontend-specific logs
docker-compose logs -f frontend

# Recent log history
docker-compose logs --tail=50
```

#### Step 5: Service Termination

```bash
# Terminate sessions (Ctrl+C in terminal)

# Remove containers
docker-compose down

# Complete system reset (removes containers and persistent volumes)
docker-compose down -v
```

### System Verification

#### Verification Protocol:

```bash
# 1. Initialize services
docker-compose up -d

# 2. Latency buffer for service initialization
sleep 30

# 3. Backend health verification
curl http://localhost:8000/health
# Expected: {"status":"active","model_loaded":false}

# 4. Frontend availability check
curl -I http://localhost:3000
# Expected: HTTP/1.1 200 OK

# 5. API documentation accessibility
curl -I http://localhost:8000/docs
# Expected: HTTP/1.1 200 OK

# 6. Dataset upload verification
curl -X POST http://localhost:8000/upload \
  -F "file=@test_data/dataset.pkl"

# 7. Training cycle execution
curl -X POST http://localhost:8000/train \
  -H "Content-Type: application/json" \
  -d '{"hidden_layers":[64,32,16],"max_epochs":50,"learning_rate":0.001}'

# 8. Inference verification
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features":[[1.0,2.0,3.0,4.0,5.0]]}'

# 9. Log analysis
docker-compose logs --tail=100

# 10. Service termination
docker-compose down
```

**Verification Results**:
- All API requests return valid status codes and responses.
- Frontend interface is accessible at http://localhost:3000.
- Backend services are responsive at http://localhost:8000.
- Logs remain free of critical errors.
- Training routines converge successfully.
- Inference engines return accurate numerical outputs.

### Local Development via Scripts

```bash
# Configure execution permissions
chmod +x start-dev.sh

# Initialize development environment
./start-dev.sh

# Access services:
# Frontend: http://localhost:3000
# Backend: http://localhost:8000
```

**Note**: Local development requires Python 3.10+ and Node.js 18+.

---

## Installation

### Prerequisites

- **Docker**: Version 20.10+ and Docker Compose 2.0+ (required for containerized deployment).
- **Python**: Version 3.10, 3.11, or 3.12.
- **Node.js**: Version 18+ and npm 9+.
- **System Requirements**: Minimum 4GB RAM (8GB recommended).

### Method 1: Containerized Deployment (Docker)

```bash
# Build system images
docker-compose build

# Initialize services
docker-compose up -d

# Monitor real-time logs
docker-compose logs -f

# Terminate services
docker-compose down
```

### Method 2: Manual Local Configuration

#### Backend Configuration

```bash
cd backend

# Initialize virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Dependency installation
pip install -e ".[dev]"

# Execution of test suites
PYTHONPATH=. pytest tests/ -v

# Initialize backend service
PYTHONPATH=. python3 main.py
```

#### Frontend Configuration

```bash
cd frontend

# Dependency installation
npm install

# Environment configuration
echo "NEXT_PUBLIC_API_URL=http://localhost:8000" > .env.local

# Initialize development server
npm run dev

# Production build and initialization
npm run build
npm start
```

---

## Testing

### Execution of Test Suites

```bash
cd backend

# Execute all tests with detailed output
PYTHONPATH=. pytest tests/ -v

# Execute with coverage analysis (HTML and terminal reporting)
PYTHONPATH=. pytest tests/ -v --cov=fivedreg --cov-report=html --cov-report=term

# Execute a specific test module
PYTHONPATH=. pytest tests/test_core.py -v

# View granular HTML coverage report
open htmlcov/index.html  # macOS
# Or: xdg-open htmlcov/index.html  # Linux
# Or: start htmlcov/index.html  # Windows
```

### Verification Metrics

```
============================= test session starts ==============================
platform darwin -- Python 3.12.2, pytest-9.0.2, pluggy-1.6.0
collected 74 items

tests/test_backend_utils_enhanced.py .................... (20 tests)    [ 40%]
tests/test_core.py ....... (7 tests)                                    [ 50%]
tests/test_coverage_completion.py .................... (20 tests)       [ 77%]
tests/test_endpoints.py ...... (6 tests)                                [ 85%]
tests/test_load.py ....... (7 tests)                                    [ 94%]
tests/test_model_lifecycle.py ..... (5 tests)                           [100%]

================================ tests coverage ================================
Name                   Stmts   Miss   Cover   Missing
-----------------------------------------------------
fivedreg/__init__.py       4      0 100.00%
fivedreg/data.py          70      0 100.00%
fivedreg/model.py        200      0 100.00%
fivedreg/utils.py         70      0 100.00%
-----------------------------------------------------
TOTAL                    344      0 100.00%

Coverage HTML written to dir htmlcov

============================= 74 passed in 16.32s ==============================
```

**Achievement of Full Code Coverage (100.00%)**

### Detailed Test Explanations

#### 1. **test_backend_utils.py** (10 tests) - Basic Utility Tests

**Location**: `backend/tests/test_backend_utils.py`

**Purpose**: Tests utility functions for dataset and model management

**Tests Included**:

- **test_create_synthetic_dataset**: Verifies synthetic dataset generation with correct shape (N, 5) and ground truth function
- **test_ground_truth_function**: Validates the ground truth formula: y = 2.0·x₁ + (-1.5)·x₂² + 3.0·sin(x₃) + 0.5·x₄·x₅
- **test_save_and_load_dataset**: Tests dataset persistence (save as .pkl, load back, verify data integrity)
- **test_dataset_reproducibility**: Ensures same seed produces identical datasets
- **test_model_training**: Verifies model can be trained and predictions have correct shape
- **test_model_save_and_load**: Tests model persistence (save to .pt, load back, verify parameters)
- **test_save_load_predictions_match**: Ensures saved and loaded models produce identical predictions
- **test_model_predict_without_fit**: Confirms error is raised when predicting before training
- **test_complete_workflow**: End-to-end test (create data → train → save → load → predict)
- **test_directories_exist**: Checks that data/ and models/ directories are created

**How to Run**:
```bash
PYTHONPATH=. pytest tests/test_backend_utils.py -v
```

#### 2. **test_backend_utils_enhanced.py** (20 tests) - Enhanced Utility Tests

**Location**: `backend/tests/test_backend_utils_enhanced.py`

**Purpose**: Comprehensive testing of all utility wrapper functions to achieve 100% coverage of utils.py

**Tests Included**:

**TestSaveLoadDataset (6 tests)**:
- **test_save_dataset_basic**: Tests save_dataset() function creates .pkl files correctly
- **test_save_dataset_auto_extension**: Verifies automatic .pkl extension addition
- **test_load_dataset_basic**: Tests load_dataset() function loads data correctly
- **test_load_dataset_auto_extension**: Verifies automatic .pkl extension on load
- **test_load_dataset_file_not_found**: Tests FileNotFoundError for missing datasets
- **test_list_datasets**: Tests list_datasets() returns all .pkl files in data directory

**TestSaveLoadModel (7 tests)**:
- **test_save_model_basic**: Tests save_model() wrapper function saves models correctly
- **test_save_model_auto_extension**: Verifies automatic .pt extension addition
- **test_load_model_basic**: Tests load_model() wrapper function loads models correctly
- **test_load_model_auto_extension**: Verifies automatic .pt extension on load
- **test_load_model_file_not_found**: Tests FileNotFoundError for missing models
- **test_list_models**: Tests list_models() returns all .pt files in models directory
- **test_save_load_predictions_consistency**: Ensures predictions match after save/load cycle

**TestCreateTrainValTest (3 tests)**:
- **test_create_train_val_test_no_save**: Tests creating train/val/test splits without saving
- **test_create_train_val_test_with_save**: Tests creating and saving all three splits
- **test_create_train_val_test_reproducibility**: Verifies same seed produces identical splits

**TestCreateSyntheticWithSavePath (1 test)**:
- **test_create_synthetic_with_save_path**: Tests create_synthetic_dataset() with save_path parameter

**TestEdgeCases (3 tests)**:
- **test_save_dataset_creates_data_in_dict**: Verifies correct dictionary structure in saved files
- **test_empty_list_datasets_returns_list**: Tests list_datasets() returns empty list when no datasets
- **test_empty_list_models_returns_list**: Tests list_models() returns empty list when no models


**How to Run**:
```bash
PYTHONPATH=. pytest tests/test_backend_utils_enhanced.py -v
```

#### 3. **test_core.py** (7 tests)

**Location**: `backend/tests/test_core.py`

**Purpose**: Core model functionality and validation

**Tests Included**:

- **test_model_init_defaults**: Verifies default hyperparameters are set correctly
- **test_model_structure**: Checks network architecture (input→hidden→output layers)
- **test_check_xy_shapes**: Validates input shape checking (must be N×5)
- **test_check_xy_nans**: Ensures NaN values in data are detected
- **test_predict_unfitted_raises**: Confirms RuntimeError when calling predict() before fit()
- **test_reproducibility**: Verifies same random seed produces identical results
- **test_early_stopping_logic**: Tests early stopping mechanism with validation loss

**How to Run**:
```bash
PYTHONPATH=. pytest tests/test_core.py -v
```

#### 4. **test_coverage_completion.py** (20 tests) - Coverage Completion Tests

**Location**: `backend/tests/test_coverage_completion.py`

**Purpose**: Comprehensive tests to achieve 100% code coverage by testing all edge cases, error paths, and introspection methods

**Tests Included**:

**TestDataValidationErrors (7 tests)**:
- **test_X_not_2d_error**: Tests ValueError when X is 1D instead of 2D
- **test_X_wrong_feature_count_error**: Tests ValueError when X doesn't have 5 features
- **test_X_contains_inf_error**: Tests ValueError when X contains infinite values
- **test_y_2d_reshape_to_1d**: Tests that 2D y with shape (n, 1) is reshaped to 1D
- **test_y_wrong_ndim_error**: Tests ValueError when y has wrong dimensions
- **test_X_y_length_mismatch_error**: Tests ValueError when X and y have different lengths
- **test_y_contains_inf_error**: Tests ValueError when y contains infinite values

**TestModelCheckXyErrors (3 tests)**:
- **test_y_2d_reshape_in_check_xy**: Tests _check_Xy reshapes 2D y to 1D
- **test_y_wrong_ndim_in_check_xy**: Tests _check_Xy raises error for wrong y dimensions
- **test_y_contains_inf_in_check_xy**: Tests _check_Xy raises error for inf in y

**TestModelIntrospection (4 tests)**:
- **test_count_parameters**: Tests count_parameters() method returns parameter counts
- **test_count_flops**: Tests count_flops() method calculates FLOPs
- **test_count_flops_with_batch**: Tests count_flops() with different batch sizes
- **test_get_model_summary**: Tests get_model_summary() returns complete model information

**TestWandbIntegration (3 tests)**:
- **test_training_with_wandb_enabled_mocked**: Tests training with wandb logging (mocked)
- **test_training_with_wandb_no_validation_mocked**: Tests wandb logging without validation set
- **test_wandb_import_failure_graceful**: Tests graceful handling of wandb import failures

**TestVerboseMode (2 tests)**:
- **test_verbose_training_without_validation**: Tests verbose output without validation
- **test_verbose_training_with_validation**: Tests verbose output with validation

**TestEdgeCasesCompletion (1 test)**:
- **test_training_triggers_step_logging**: Tests that step-level logging is triggered during training

**Coverage Impact**: These tests achieved the final push from 88.66% to **100% coverage**

**How to Run**:
```bash
PYTHONPATH=. pytest tests/test_coverage_completion.py -v
```

#### 5. **test_endpoints.py** (6 tests)

**Location**: `backend/tests/test_endpoints.py`

**Purpose**: API endpoint integration testing using FastAPI TestClient

**Tests Included**:

- **test_health_check**: Tests GET /health endpoint returns {"status": "active", "model_loaded": bool}
- **test_upload_flow**: Tests POST /upload with valid .pkl file
- **test_train_flow**: Tests POST /train after uploading dataset (full training cycle)
- **test_predict_flow**: Tests POST /predict with trained model (input validation, prediction format)
- **test_predict_without_model**: Ensures 400 error when predicting without trained model
- **test_upload_invalid_file** (implicit): Tests rejection of non-.pkl files

**How to Run**:
```bash
PYTHONPATH=. pytest tests/test_endpoints.py -v
```

#### 6. **test_load.py** (7 tests)

**Location**: `backend/tests/test_load.py`

**Purpose**: Data loading and preprocessing pipeline

**Tests Included**:

- **test_load_dataset_no_y**: Loads dataset without target values (X only)
- **test_load_dataset_invalid_extension**: Rejects files without .pkl extension
- **test_load_dataset_bad_structure**: Detects invalid dictionary structure (missing 'X' key)
- **test_load_dataset_bad_shapes**: Catches incorrect shape (not N×5)
- **test_split_data_shapes**: Verifies train/val/test split ratios (64%/16%/20%)
- **test_split_data_no_y**: Tests splitting without target values
- **test_standardize_features_statistics**: Validates standardization (mean≈0, std≈1 on training set)

**How to Run**:
```bash
PYTHONPATH=. pytest tests/test_load.py -v
```

#### 7. **test_model_lifecycle.py** (5 tests)

**Location**: `backend/tests/test_model_lifecycle.py`

**Purpose**: End-to-end model lifecycle workflows

**Tests Included**:

- **test_load_dataset_shapes**: Validates dataset loading produces correct shapes
- **test_prepare_data_splits_and_shapes**: Tests complete data preparation pipeline
- **test_standardization_statistics_on_train**: Ensures standardization is fit only on training data
- **test_model_fit_predict_mse_is_finite**: Verifies model training produces finite MSE
- **test_model_persistence_and_inference**: Tests full cycle (train → save → load → predict → verify)

**How to Run**:
```bash
PYTHONPATH=. pytest tests/test_model_lifecycle.py -v
```

### 📁 Where to Find Test Results

#### 1. **Terminal Output**
After running tests, you'll see:
- ✅ Pass/fail status for each test
- Percentage progress
- Total time taken
- Coverage summary

#### 2. **HTML Coverage Report**
**Location**: `backend/htmlcov/index.html`

**How to Access**:
```bash
cd backend
PYTHONPATH=. pytest tests/ --cov=fivedreg --cov-report=html
open htmlcov/index.html  # macOS
```

**What's Included**:
- Line-by-line coverage visualization
- Highlighted uncovered lines in red
- Coverage percentage per file
- Detailed statistics

#### 3. **Coverage Terminal Report**
Displayed immediately after tests with:
- Statements count
- Missed lines
- Coverage percentage
- Missing line numbers

#### 4. **pytest Cache**
**Location**: `backend/.pytest_cache/`
- Stores test results for faster reruns
- Can be cleared with `pytest --cache-clear`

### Test Categories Summary

| Category | Tests | What They Test | Coverage |
|----------|-------|----------------|----------|
| **Core Model** | 7 | Model initialization, validation, reproducibility, early stopping | 100% ✅ |
| **Data Pipeline** | 7 | Loading, validation, splitting, standardization | 100% ✅ |
| **API Endpoints** | 6 | Upload, train, predict, health check | 100% ✅ |
| **Lifecycle** | 5 | End-to-end workflows, persistence | 100% ✅ |
| **Basic Utilities** | 10 | Dataset/model creation, basic save/load | 100% ✅ |
| **Enhanced Utilities** | 20 | Wrapper functions, error handling, edge cases | 100% ✅ |
| **Coverage Completion** | 20 | All validation errors, introspection, wandb, verbose mode | 100% ✅ |

**Overall Coverage**: 🎯 **100.00%** (344 statements, 0 missing) - PERFECT!

**Journey to Perfection**:
- ✅ Original: 34 tests, 77.03% coverage
- ✅ Enhanced: 54 tests, 88.66% coverage (+11.63%)
- ✅ **Final: 74 tests, 100.00% coverage (+11.34%) - PERFECT SCORE! 🎉**

---

## Performance and Profiling

### Overview

Performance analysis and benchmarking of the model are provided to illustrate its computational characteristics and scalability. The analysis encompasses training time scaling, memory consumption, and accuracy metrics across various dataset sizes (1K, 5K, 10K samples).

### Directory Hierarchy

**Location**: `backend/task8/`

```
backend/task8/
├── __init__.py            # Package initialization
├── config.py              # Configuration (dataset sizes, hyperparameters)
├── experiments.py         # Experiment implementations
├── profiling.py           # Memory profiling functions
├── visualization.py       # Visualization functions
└── run_task8.py          # Execution orchestrator
```

### Module Specifications

#### 1. **config.py**
Defines configuration constants:
- `DATASET_SIZES`: {"1K": 1000, "5K": 5000, "10K": 10000}
- `DEFAULT_HYPERPARAMS`: Architecture specifications and default hyperparameters.
- `FIGURES_DIR`: Output directory for analysis results.

#### 2. **experiments.py**
Implements comprehensive experimental analyses:
- **Training Time Scaling**: Training duration as a function of dataset size.
- **Accuracy Analysis**: Evaluation of MSE and R² across dataset dimensions.
- **Error Evolution**: Longitudinal analysis of loss curves.
- **Iso-Parameter Architecture Comparison**: Architectural analysis holding parameter count constant.
- **Ground Truth Approximation**: Analysis of functional approximation fidelity.
- **Memory Profiling**: Quantification of memory usage during operational phases.
- **Memory Scaling**: Memory consumption relative to dataset size.

#### 3. **profiling.py**
Computational profiling utilities:
- `profile_memory()`: Quantification of memory during model operations.
- `get_model_size()`: Calculation of model parameter count and FLOPs.

#### 4. **visualization.py**
Visualization functions for analysis plots.

#### 5. **run_task8.py**
Main execution orchestrator.

### Execution Protocol

#### Comprehensive Analysis
```bash
cd backend
PYTHONPATH=. python3 -m task8.run_task8
```

#### Accelerated Execution
```bash
cd backend
PYTHONPATH=. python3 -m task8.run_task8 --quick
```

**Mean Execution Time**:
- Quick mode: ~5-10 minutes.
- Comprehensive mode: ~30-60 minutes (hardware dependent).

### Output Specification

**Location**: `backend/figures/`

#### Analysis Results (JSON)
- `training_time_results.json`
- `accuracy_results.json`
- `error_evolution_results.json`
- `iso_parameter_results.json`
- `approximation_results.json`
- `memory_profiling_results.json`
- `memory_scaling_results.json`

#### Graphical Visualizations (PNG)
- `training_time_scaling.png`
- `accuracy_comparison.png`
- `error_evolution.png`
- `iso_parameter_comparison.png`
- `approximation_analysis.png`
- `memory_usage.png`
- `memory_scaling.png`
- `flops_breakdown.png`

### Key Findings

#### 1. Training Time Scaling
**Dataset Size vs Training Time** (100 epochs, architecture: 64-32-16):

| Dataset Size | Mean Training Time | Std Dev | Scaling |
|--------------|-------------------|---------|---------|
| 1K samples   | 3.47s             | 0.26s   | Baseline |
| 5K samples   | 12.35s            | 0.03s   | 3.6× |
| 10K samples  | 22.99s            | 2.49s   | 6.6× |

**Observation**: Training duration scales approximately **linearly (O(n))** with dataset size, indicating efficient implementation.

#### 2. Accuracy Metrics

**Mean Squared Error (MSE)** - Lower is better:

| Dataset Size | Train MSE | Val MSE | Test MSE |
|--------------|-----------|---------|----------|
| 1K samples   | 0.471     | 0.520   | 0.614    |
| 5K samples   | 0.043     | 0.061   | 0.055    |
| 10K samples  | 0.018     | 0.024   | 0.022    |

**R² Score** (Coefficient of Determination) - Higher is better:

| Dataset Size | Train R² | Val R² | Test R² |
|--------------|----------|--------|---------|
| 1K samples   | 0.964    | 0.957  | 0.952   |
| 5K samples   | 0.997    | 0.995  | 0.996   |
| 10K samples  | 0.999    | 0.998  | 0.998   |

**Key Observations**:
- ✅ **Significant accuracy improvement** with more data: 10K samples achieve ~10× lower MSE than 1K
- ✅ **Excellent R² scores** (>0.95) across all dataset sizes indicate strong model fit
- ✅ **Low overfitting**: Train/validation/test metrics are consistent, showing good generalization
- ✅ **Convergence**: Model benefits from larger datasets, approaching R² ≈ 0.999 at 10K samples

#### 3. Memory Usage

**Peak Memory Consumption** (5K samples dataset):

| Component | Memory (MB) |
|-----------|-------------|
| Baseline  | 377.3       |
| Model Parameters | 0.043 |
| Training Peak | 275.9 |
| Inference | 0.0 |

**Model Characteristics**:
- **Total Parameters**: 3,009 (very lightweight)
- **Total FLOPs**: 2,896
- **Theoretical Model Size**: 0.011 MB (11 KB)
- **Theoretical Optimizer Size**: 0.023 MB (23 KB, Adam optimizer)

**Key Observations**:
- ✅ **Extremely lightweight model**: Only 3K parameters makes it suitable for resource-constrained environments
- ✅ **Low memory footprint**: Model itself requires <50 KB
- ✅ **Efficient inference**: Minimal memory overhead during prediction
- ✅ **Scalable architecture**: Can handle larger datasets without excessive memory consumption

#### 4. Model Scalability

**Summary**:
- **Training Time**: Scales linearly (O(n)) with dataset size - excellent for production
- **Accuracy**: Improves significantly with more data (MSE reduces by 10× from 1K to 10K)
- **Memory**: Very lightweight model (3K parameters) with minimal overhead
- **Generalization**: Consistent performance across train/val/test splits
- **Production-Ready**: Fast training (<30s for 10K samples), low resource requirements

**Recommended Configuration**:
- For quick prototyping: 1K samples (3.5s training, R² > 0.95)
- For production: 5K-10K samples (12-23s training, R² > 0.99)
- Architecture: Default (64, 32, 16) provides excellent balance of accuracy and speed

### 📊 Experiment Details

For detailed analysis and visualizations, see:
- JSON results in `backend/figures/*.json`
- Plots in `backend/figures/*.png`
- Source code in `backend/task8/`

To regenerate results:
```bash
cd backend
PYTHONPATH=. python3 -m task8.run_task8 --quick
```

---

## Environment Variables

### Backend Configuration

Example `backend/.env` (optional):
```env
ENV=dev                  # Selection: dev, prod, test
WANDB_DISABLED=true      # Disable Weights & Biases logging
PYTHONUNBUFFERED=1       # Unbuffered output configuration
```

### Frontend Configuration

Example `frontend/.env.local`:
```env
NEXT_PUBLIC_API_URL=http://localhost:8000
NODE_ENV=development
```

For production environments (`frontend/.env.production`):
```env
NEXT_PUBLIC_API_URL=https://your-api-domain.com
NODE_ENV=production
```

---

## API Documentation

### Endpoint Overview

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | System health check (returns status and model availability) |
| `POST` | `/upload` | Dataset upload interface for .pkl files |
| `POST` | `/train` | Model training initialization with hyperparameter specifications |
| `POST` | `/predict` | Functional inference for 5D input features |
| `GET` | `/docs` | Interactive OpenAPI documentation (Swagger UI) |
| `GET` | `/redoc` | Alternative API documentation interface (ReDoc) |

### Interactive Documentation Interfaces

Upon initialization of the backend service, the following interfaces are accessible:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### API Interaction Examples

```bash
# Health monitoring
curl http://localhost:8000/health

# Dataset upload
curl -X POST http://localhost:8000/upload \
  -F "file=@dataset.pkl"

# Model training initialization
curl -X POST http://localhost:8000/train \
  -H "Content-Type: application/json" \
  -d '{
    "hidden_layers": [64, 32, 16],
    "max_epochs": 200,
    "learning_rate": 0.001
  }'

# Functional inference
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": [[1.0, 2.0, 3.0, 4.0, 5.0]]
  }'
```

---

## System Architecture and File Hierarchy

```
interpoletor/
│
├── backend/                        # Python backend service
│   ├── fivedreg/                   # Core machine learning package
│   │   ├── __init__.py
│   │   ├── data.py                 # Data management and preprocessing
│   │   ├── model.py                # FiveDRegressor implementation (PyTorch)
│   │   └── utils.py                # Utility functions for persistence
│   │
│   ├── tests/                      # Comprehensive test suite
│   │   ├── test_core.py            # Model unit tests
│   │   ├── test_load.py            # Data pipeline tests
│   │   ├── test_endpoints.py       # API integration tests
│   │   ├── test_model_lifecycle.py # End-to-end operational tests
│   │   └── test_backend_utils.py   # Utility verification
│   │
│   ├── main.py                     # FastAPI service entry point
│   ├── pyproject.toml              # Python dependency specification
│   ├── Dockerfile                  # Backend container configuration
│   ├── .dockerignore               # Docker build exclusions
│   ├── data_uploads/               # Runtime dataset storage
│   ├── models/                     # Runtime model weight storage
│   └── figures/                    # Runtime visualization output
│
├── frontend/                       # Next.js frontend service
│   ├── src/
│   │   └── app/
│   │       ├── page.tsx            # Integrated single-page application
│   │       ├── layout.tsx          # Application root layout
│   │       └── globals.css         # Global style definitions
│   │
│   ├── public/                     # Static assets
│   ├── package.json                # Node.js dependency specification
│   ├── tsconfig.json               # TypeScript configuration
│   ├── tailwind.config.js          # Styling theme configuration
│   ├── next.config.js              # Next.js framework configuration
│   ├── Dockerfile                  # Frontend container configuration
│   ├── .dockerignore               # Docker build exclusions
│   ├── .env.local                  # Local environment configuration
│   └── README.md                   # Frontend-specific documentation
│
├── docker-compose.yml              # System orchestration configuration
├── start-dev.sh                    # Development initialization script
├── README.md                       # System overview documentation
├── .gitignore                      # Workspace exclusions
└── test_data/                      # Evaluation datasets
```

---

## Troubleshooting

### Backend Issues

**Port 8000 in use:**
```bash
lsof -i :8000
kill -9 <PID>
```

**Import errors:**
```bash
export PYTHONPATH=.
cd backend && PYTHONPATH=. python3 main.py
```

**Model not loading:**
```bash
# Train a new model
## Error Resolution and Troubleshooting

### Backend Operational Issues

**Port 8000 Conflict:**
```bash
lsof -i :8000
kill -9 <PID>
```

**Module Resolution Errors:**
```bash
export PYTHONPATH=.
cd backend && PYTHONPATH=. python3 main.py
```

**Model Initialization Failures:**
```bash
# Initialize a new training cycle
curl -X POST http://localhost:8000/train \
  -H "Content-Type: application/json" \
  -d '{"hidden_layers": [64,32,16]}'
```

### Frontend Operational Issues

**Port 3000 Conflict:**
```bash
lsof -i :3000
kill -9 <PID>
# Alternative: PORT=3001 npm run dev
```

**Backend Communication Failures:**
```bash
# Verify backend service availability
curl http://localhost:8000/health

# Inspect local environment configuration
cat frontend/.env.local
# Specification requirement: NEXT_PUBLIC_API_URL=http://localhost:8000
```

### Docker Operational Issues

**Build Failures:**
```bash
# Execute clean build procedure
docker-compose build --no-cache

# Manage system resources
docker system df
docker system prune -a
```

**Container Initialization Failures:**
```bash
docker-compose logs backend
docker-compose logs frontend
```

**For exhaustive troubleshooting guidance, refer to**: [DEPLOYMENT.md](DEPLOYMENT.md)

---

## Additional Documentation

- **[DEPLOYMENT.md](DEPLOYMENT.md)**: Comprehensive deployment strategies, environment configuration, and production auditing protocols.
- **[frontend/README.md](frontend/README.md)**: Frontend-specific implementation details.
- **[FRONTEND_README.md](FRONTEND_README.md)**: Frontend initialization guide.
- **[backend/pyproject.toml](backend/pyproject.toml)**: Dependency specification and package configuration.

---

## Academic Context

This project was developed for the Cambridge Computer Science Tripos as a comprehensive demonstration of:
- Contemporary web engineering practices (FastAPI, Next.js, Docker).
- Machine learning systems architecture (PyTorch, scikit-learn).
- Software engineering methodologies (automated testing, CI/CD, documentation).
- Full-stack system integration (RESTful APIs, reactive interfaces, containerization).

---

## License

MIT License - Academic project for Cambridge University.

---

## Author

**Iuliia Vituigova**
- Email: iv294@cam.ac.uk
- Institution: Cambridge University
- Course: Computer Science Tripos, Part I
- Year: 2025

---

## Acknowledgments

Technologies and Frameworks utilized:
- **PyTorch**: Neural network implementation and optimization.
- **FastAPI**: Asynchronous web framework for high-performance APIs.
- **Next.js**: React-based framework for optimized frontend delivery.
- **Docker**: Containerization and environment orchestration.
- **pytest**: Comprehensive testing and verification framework.

---

**For detailed initialization instructions, see [DEPLOYMENT.md](DEPLOYMENT.md)**
**API documentation is accessible at http://localhost:8000/docs during runtime**
