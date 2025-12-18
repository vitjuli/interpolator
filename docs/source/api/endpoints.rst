REST API Endpoints
==================

The backend exposes a RESTful API built with FastAPI.

Base URL
--------

When running locally: ``http://localhost:8000``

Interactive Documentation
--------------------------

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI Schema**: http://localhost:8000/openapi.json

Endpoints
---------

Health Check
~~~~~~~~~~~~

**GET** ``/health``

Check API status.

.. code-block:: bash

   curl http://localhost:8000/health

Response:

.. code-block:: json

   {
     "status": "active",
     "model_loaded": false
   }

Upload Dataset
~~~~~~~~~~~~~~

**POST** ``/upload``

Upload a pickle dataset file.

.. code-block:: bash

   curl -X POST http://localhost:8000/upload \
     -F "file=@dataset.pkl"

Train Model
~~~~~~~~~~~

**POST** ``/train``

Train a new model.

.. code-block:: bash

   curl -X POST http://localhost:8000/train \
     -H "Content-Type: application/json" \
     -d '{
       "hidden_layers": [64, 32, 16],
       "max_epochs": 100,
       "learning_rate": 0.001
     }'

Make Prediction
~~~~~~~~~~~~~~~

**POST** ``/predict``

Make predictions on new data.

.. code-block:: bash

   curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d '{
       "features": [[1.0, 2.0, 3.0, 4.0, 5.0]]
     }'
