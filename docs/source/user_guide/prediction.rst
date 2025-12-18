Making Predictions
==================

This guide explains how to make predictions with trained models.

Prerequisites
-------------

- Model must be trained first
- Input must be 5 features

Via Web Interface
-----------------

1. Ensure model is trained
2. Go to "Make Predictions" section
3. Enter 5 feature values
4. Click "Predict"
5. View result

Via API
-------

.. code-block:: python

   import requests

   features = [[1.0, 2.0, 3.0, 4.0, 5.0]]

   response = requests.post(
       'http://localhost:8000/predict',
       json={'features': features}
   )

   result = response.json()
   print(f"Prediction: {result['predictions'][0]}")

Batch Predictions
-----------------

.. code-block:: python

   # Predict multiple samples
   features = [
       [1.0, 2.0, 3.0, 4.0, 5.0],
       [2.0, 3.0, 4.0, 5.0, 6.0],
       [3.0, 4.0, 5.0, 6.0, 7.0]
   ]

   response = requests.post(
       'http://localhost:8000/predict',
       json={'features': features}
   )

   predictions = response.json()['predictions']
