Benchmarks
==========

System Performance Benchmarks
------------------------------

Training Performance
~~~~~~~~~~~~~~~~~~~~

- **10K samples**: 20.91 ± 5.49 seconds
- **Throughput**: ~478 samples/second
- **Epochs**: 100 (with early stopping)

Inference Performance
~~~~~~~~~~~~~~~~~~~~~

- **Latency**: <1ms per prediction
- **Batch size 32**: ~30ms
- **Batch size 256**: ~100ms

Model Characteristics
~~~~~~~~~~~~~~~~~~~~~

- **Parameters**: 3,009
- **Model size**: 12 KB
- **Memory (training)**: 234 MB peak
- **Memory (inference)**: 0.05 MB

Scaling Predictions
-------------------

Based on observed linear scaling:

.. list-table::
   :header-rows: 1

   * - Dataset Size
     - Estimated Time
     - Estimated Memory
   * - 50K
     - ~1.7 minutes
     - ~400 MB
   * - 100K
     - ~3.5 minutes
     - ~550 MB
   * - 1M
     - ~35 minutes
     - ~1.5 GB

Hardware Requirements
---------------------

**Minimum**:
- RAM: 512 MB
- CPU: 1 core
- Disk: 100 MB

**Recommended**:
- RAM: 2 GB
- CPU: 2-4 cores
- Disk: 500 MB

**Optimal**:
- RAM: 4 GB
- CPU: 4+ cores
- GPU: Optional (2-3× speedup)
