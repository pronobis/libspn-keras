Performance Test Results
========================

- with_gpu_bounds_check - the GPU kernel of custom ops still had bounds check enabled
- without_gpu_bounds_check - the GPU kernel now does not have the bounds check
- with_compute_capabilities - the version without bounds check is teseted after we added compilation specifically for compute capabilities  ["3.5", "5.2", "6.1"] to setup.py to fix the error on GTX1050.
