Data Module API
===============

The ``fivedreg.data`` module handles dataset loading, validation, and preprocessing.

.. automodule:: fivedreg.data
   :members:
   :undoc-members:
   :show-inheritance:

Key Functions
-------------

load_dataset
~~~~~~~~~~~~

.. autofunction:: fivedreg.data.load_dataset

create_synthetic_dataset
~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: fivedreg.data.create_synthetic_dataset

ground_truth_function
~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: fivedreg.data.ground_truth_function

Usage Example
-------------

.. code-block:: python

   from fivedreg.data import load_dataset, create_synthetic_dataset

   # Create synthetic data
   X, y = create_synthetic_dataset(n_samples=1000, seed=42)

   # Load from file
   X_train, y_train, X_val, y_val, X_test, y_test = load_dataset(
       'dataset.pkl',
       test_size=0.2,
       val_size=0.1
   )
