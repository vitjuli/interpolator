Training Models
===============

This guide covers model training in detail.

Basic Training
--------------

Via Web Interface
~~~~~~~~~~~~~~~~~

1. Navigate to http://localhost:3000
2. Upload your dataset
3. Configure architecture (or use defaults)
4. Click "Train Model"
5. Monitor progress in real-time

Via API
~~~~~~~

.. code-block:: python

   import requests

   config = {
       'hidden_layers': [64, 32, 16],
       'max_epochs': 100,
       'learning_rate': 0.001
   }

   response = requests.post(
       'http://localhost:8000/train',
       json=config
   )

   result = response.json()
   print(f"Final R²: {result['test_r2']}")

Advanced Configuration
----------------------

Architecture Selection
~~~~~~~~~~~~~~~~~~~~~~

Choose architecture based on dataset size:

- **Small (1K)**: ``[32, 16]``
- **Medium (5K)**: ``[64, 32, 16]``
- **Large (10K+)**: ``[128, 64, 32]``

Hyperparameter Tuning
~~~~~~~~~~~~~~~~~~~~~~

Key hyperparameters:

- ``learning_rate``: 0.0001 to 0.01
- ``batch_size``: 64 to 512
- ``patience``: 10 to 30 epochs
- ``max_epochs``: 50 to 200

Best Practices
--------------

1. Use validation split (10-20%)
2. Enable early stopping
3. Start with default hyperparameters
4. Monitor training/validation loss
5. Save best model checkpoint
