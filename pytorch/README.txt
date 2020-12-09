Implementation of ConvTranspose2d in PyTorch (C++ and Python)
====================================================================================

- Instructions to run C++ implementation on CIMS CUDA Cluster (tested on cuda4):

# load modules
module load cmake-3
module load gcc-9.2
module load cuda-10.2

# download Pytorch C++ API
wget https://download.pytorch.org/libtorch/cu102/libtorch-shared-with-deps-1.7.0.zip
unzip libtorch-shared-with-deps-1.7.0.zip

# build
mkdir build && cd build 
cmake -DCMAKE_CXX_COMPILER=g++-9.2 -DCMAKE_C_COMPILER=gcc-9.2 -DCMAKE_PREFIX_PATH=libtorch ..

# compile
cmake --build .

# run
./pytorch

====================================================================================

- Instructions to run Python implementation on CIMS CUDA Cluster:

module load cuda-10.2
python3 pytorch.py
