# Dense-Matrix-Multiplication-in-CUDA

### Testing Hardware

Performance testing executed on an NVIDIA GeForce RTX 3050 Laptop GPU

NVIDIA GeForce RTX 3050 Laptop GPU Specs:

Theoretical Performance (FP32) - 5.501 TFLOPS
Global Memory Size - 4 GB
Global Memory Bandwith - 192.0 GB/s
Streaming Multiprocessor Count - 16
Shared Memory Size - 128 KB per Streaming Multiprocessor

### Performance Tests

Cublas Performance (8192 x 8192):

3309.71 GFlops/s
332.208 ms

Naive Algorithm Performance (8192 x 8192):

54.01 GFlops/s
20357.662 ms

Global Memory Thread Coelesce (8192 x 8192):

363.48 GFlops/s
3024.940 ms

Shared Memory Cache Blocking (8192 x 8192):

485.80 GFlops/s
2263.282 ms