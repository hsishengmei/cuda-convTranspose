Implementation of ConvTranspose2d in CUDA
====================================================================================

# Compile
nvcc -o cudaConvTranspose CudaConvTranspose2d.cu

# Run
./cudaConvTranspose [algo] [layer]

algo: 0, 1, 2, 3, 4
- algo 0: cpu naive implementation
- algo 1: gpu naive implementation
- algo 2: gpu share filter & ofmap, tiled (best)
- algo 3: cpu group by ofmap
- algo 4: gpu group by ofmap, share ifmap

layer: 0, 1, 2, 3
- layer 0: (W,C,M) = (4,512,256)
- layer 1: (W,C,M) = (8,256,128)
- layer 2: (W,C,M) = (16,128,64)
- layer 3: (W,C,M) = (32,64,3)
