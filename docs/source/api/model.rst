Model Module API
================

The ``fivedreg.model`` module contains the FiveDRegressor neural network implementation.

.. automodule:: fivedreg.model
   :members:
   :undoc-members:
   :show-inheritance:

FiveDRegressor Class
--------------------

.. autoclass:: fivedreg.model.FiveDRegressor
   :members:
   :undoc-members:
   :show-inheritance:

Key Methods
-----------

fit
~~~

Train the model on dataset.

.. automethod:: fivedreg.model.FiveDRegressor.fit

predict
~~~~~~~

Make predictions on new data.

.. automethod:: fivedreg.model.FiveDRegressor.predict

evaluate
~~~~~~~~

Evaluate model performance.

.. automethod:: fivedreg.model.FiveDRegressor.evaluate

save / load
~~~~~~~~~~~

Persist and restore models.

.. automethod:: fivedreg.model.FiveDRegressor.save
.. automethod:: fivedreg.model.FiveDRegressor.load

Usage Example
-------------

.. code-block:: python

   from fivedreg.model import FiveDRegressor
   from fivedreg.data import create_synthetic_dataset

   # Create data
   X, y = create_synthetic_dataset(1000, seed=42)

   # Initialize model
   model = FiveDRegressor(
       hidden_layers=(64, 32, 16),
       learning_rate=0.001,
       max_epochs=100
   )

   # Train
   history = model.fit(X, y)

   # Predict
   predictions = model.predict(X)

   # Save
   model.save('my_model.pt')
