Interpoletor Documentation
===========================

**Interpoletor** is a production-ready 5D regression system with comprehensive testing and performance profiling.

.. image:: https://img.shields.io/badge/coverage-100%25-brightgreen
   :alt: Test Coverage

.. image:: https://img.shields.io/badge/tests-74%20passing-success
   :alt: Tests

Quick Links
-----------

* **Frontend**: http://localhost:3000
* **Backend**: http://localhost:8000
* **API Docs**: http://localhost:8000/docs

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation
   quickstart
   deployment

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   user_guide/overview
   user_guide/training
   user_guide/prediction
   user_guide/api_usage

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/data
   api/model
   api/utils
   api/endpoints

.. toctree::
   :maxdepth: 2
   :caption: Testing

   testing/overview
   testing/test_suite
   testing/coverage

.. toctree::
   :maxdepth: 2
   :caption: Performance

   performance/profiling
   performance/full_performance_analysis
   performance/benchmarks
   performance/optimization

.. toctree::
   :maxdepth: 1
   :caption: Additional Resources

   examples
   faq
   troubleshooting
   changelog

Indices
=======

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

Quick Start
===========

Installation
------------

.. code-block:: bash

   git clone https://github.com/yourusername/interpoletor.git
   cd interpoletor
   docker-compose up --build

Access:

* **Frontend**: http://localhost:3000
* **Backend API**: http://localhost:8000/docs

Basic Usage
-----------

.. code-block:: python

   from fivedreg.model import FiveDRegressor
   from fivedreg.data import create_synthetic_dataset

   # Create data
   X, y = create_synthetic_dataset(1000, seed=42)

   # Train model
   model = FiveDRegressor(hidden_layers=(64, 32, 16))
   history = model.fit(X, y)

   # Predict
   predictions = model.predict(X)

Metrics
-----------

**Performance (10K samples)**:

- Training: 20.91 ± 5.49s
- Accuracy: R² = 0.9982 ± 0.0001
- Memory: 233.6 MB peak

**Model**:

- Parameters: 3,009 (12 KB)
- Inference: <1ms
- Architecture: Configurable

**Testing**:

- Coverage: 100% (344/344 statements)
- Tests: 74 (all passing)

