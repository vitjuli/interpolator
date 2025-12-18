Examples
========

Practical examples for common use cases.

Example 1: Basic Workflow
--------------------------

Complete workflow from data to prediction:

.. code-block:: python

   from fivedreg.data import create_synthetic_dataset
   from fivedreg.model import FiveDRegressor

   # 1. Create data
   X, y = create_synthetic_dataset(n_samples=1000, seed=42)

   # 2. Initialize model
   model = FiveDRegressor(
       hidden_layers=(64, 32, 16),
       learning_rate=0.001,
       max_epochs=100
   )

   # 3. Train
   history = model.fit(X, y)
   print(f"Final R²: {history['test_r2'][-1]:.4f}")

   # 4. Predict
   predictions = model.predict(X[:5])
   print(f"Predictions: {predictions}")

   # 5. Save model
   model.save('my_model.pt')

Example 2: Custom Dataset
--------------------------

Using your own data:

.. code-block:: python

   import numpy as np
   from fivedreg.data import load_dataset
   from fivedreg.model import FiveDRegressor

   # Load your data
   X_train, y_train, X_val, y_val, X_test, y_test = load_dataset(
       'your_dataset.pkl',
       test_size=0.2,
       val_size=0.1
   )

   # Train with validation
   model = FiveDRegressor(hidden_layers=(64, 32, 16))
   history = model.fit(X_train, y_train, X_val, y_val)

   # Evaluate
   metrics = model.evaluate(X_test, y_test)
   print(f"Test R²: {metrics['r2']:.4f}")

Example 3: Hyperparameter Search
---------------------------------

Finding optimal hyperparameters:

.. code-block:: python

   from fivedreg.data import create_synthetic_dataset
   from fivedreg.model import FiveDRegressor

   X, y = create_synthetic_dataset(1000, seed=42)

   # Try different architectures
   architectures = [
       (32, 16),
       (64, 32, 16),
       (128, 64, 32)
   ]

   best_r2 = 0
   best_arch = None

   for arch in architectures:
       model = FiveDRegressor(hidden_layers=arch, max_epochs=50)
       history = model.fit(X, y)
       r2 = history['test_r2'][-1]

       print(f"Architecture {arch}: R² = {r2:.4f}")

       if r2 > best_r2:
           best_r2 = r2
           best_arch = arch
           model.save(f'model_{arch}.pt')

   print(f"Best: {best_arch} with R² = {best_r2:.4f}")

Example 4: REST API Usage
--------------------------

Complete API workflow:

.. code-block:: python

   import requests
   import numpy as np
   import pickle

   # Create dataset
   X = np.random.randn(1000, 5)
   y = 2*X[:,0] - 1.5*X[:,1]**2

   with open('data.pkl', 'wb') as f:
       pickle.dump({'X': X, 'y': y}, f)

   # Upload
   with open('data.pkl', 'rb') as f:
       files = {'file': f}
       r = requests.post('http://localhost:8000/upload', files=files)
       print(r.json())

   # Train
   config = {'hidden_layers': [64, 32, 16], 'max_epochs': 50}
   r = requests.post('http://localhost:8000/train', json=config)
   print(f"R²: {r.json()['test_r2']}")

   # Predict
   features = [[1.0, 2.0, 3.0, 4.0, 5.0]]
   r = requests.post('http://localhost:8000/predict',
                    json={'features': features})
   print(f"Prediction: {r.json()['predictions'][0]}")

Example 5: Production Deployment
---------------------------------

Production-ready setup:

.. code-block:: python

   from fivedreg.model import FiveDRegressor
   import logging

   # Setup logging
   logging.basicConfig(level=logging.INFO)
   logger = logging.getLogger(__name__)

   # Load production model
   model = FiveDRegressor.load('production_model.pt')

   # Prediction with error handling
   def predict_safe(features):
       try:
           predictions = model.predict(features)
           logger.info(f"Predicted {len(predictions)} samples")
           return predictions
       except Exception as e:
           logger.error(f"Prediction failed: {e}")
           return None

   # Use it
   result = predict_safe([[1.0, 2.0, 3.0, 4.0, 5.0]])
   if result is not None:
       print(f"Result: {result[0]}")
