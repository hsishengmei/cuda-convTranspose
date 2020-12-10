====================================================================================
Implementation of ConvTranspose2d in CUDA
====================================================================================

# Compile
nvcc -o cudaConvTranspose CudaConvTranspose2d.cu

# Run
./cudaConvTranspose [algo] [layer]

algo: 0, 1, 2, 3, 4
- algo 0: cpu naive implementation
- algo 1: cpu group by ofmap
- algo 2: gpu naive implementation
- algo 3: gpu share filter & ofmap, tiled (best)
- algo 4: gpu group by ofmap, share ifmap

layer: 0, 1, 2, 3
- layer 0: (W,C,M) = (4,512,256)
- layer 1: (W,C,M) = (8,256,128)
- layer 2: (W,C,M) = (16,128,64)
- layer 3: (W,C,M) = (32,64,3)

====================================================================================
Comparison with ConvTranspose2d in PyTorch (C++ and Python):
====================================================================================
- Instructions to run C++ implementation on NYU CIMS CUDA Cluster (tested on cuda4):

# load modules
module load cmake-3
module load gcc-7.4
module load cuda-10.2

# download and install Pytorch C++ Library
wget https://download.pytorch.org/libtorch/cu102/libtorch-shared-with-deps-1.7.0.zip
unzip libtorch-shared-with-deps-1.7.0.zip

# build
mkdir build && cd build 
cmake -DCMAKE_CXX_COMPILER=g++-7.4 -DCMAKE_C_COMPILER=gcc-7.4 -DCMAKE_PREFIX_PATH=libtorch ..

# compile
cmake --build .

# run 
./pytorch [layer]
(layer = 0, 1, 2, 3 same as above)

====================================================================================
- Instructions to run Python implementation on NYU CIMS CUDA Cluster (tested on cuda4):

# load modules
module load cuda-10.2

# run 
python3 pytorch.py [layer]
(layer = 0, 1, 2, 3 same as above)