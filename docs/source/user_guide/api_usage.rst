API Usage Guide
===============

Complete guide to using the Interpoletor REST API.

Authentication
--------------

Currently no authentication required for local deployment.

Complete Workflow Example
--------------------------

.. code-block:: python

   import requests
   import numpy as np
   import pickle

   API_URL = "http://localhost:8000"

   # 1. Create dataset
   X = np.random.randn(1000, 5)
   y = 2.0*X[:,0] - 1.5*X[:,1]**2

   with open('data.pkl', 'wb') as f:
       pickle.dump({'X': X, 'y': y}, f)

   # 2. Upload
   with open('data.pkl', 'rb') as f:
       files = {'file': f}
       requests.post(f'{API_URL}/upload', files=files)

   # 3. Train
   config = {'hidden_layers': [64, 32, 16]}
   result = requests.post(f'{API_URL}/train', json=config)
   print(result.json())

   # 4. Predict
   features = [[1.0, 2.0, 3.0, 4.0, 5.0]]
   pred = requests.post(f'{API_URL}/predict', json={'features': features})
   print(pred.json())

Error Handling
--------------

.. code-block:: python

   try:
       response = requests.post(url, json=data)
       response.raise_for_status()
       result = response.json()
   except requests.exceptions.HTTPError as e:
       print(f"Error: {e}")
       print(response.json())
