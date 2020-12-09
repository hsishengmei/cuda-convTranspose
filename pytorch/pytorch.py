from torch import nn
import torch
import torch.nn.functional as F
import time
import sys 

K = 4

W = int(sys.argv[1])
C = int(sys.argv[2])
M = int(sys.argv[3])
    

if torch.cuda.is_available():
    print("current cuda device:", torch.cuda.current_device())

input = torch.randn(1, C, W, W, dtype=torch.float32, layout=torch.strided, requires_grad=False)
weight = torch.randn(C, M, K, K, dtype=torch.float32, layout=torch.strided, requires_grad=False)

nIter = 1000
start = time.time()

for _ in range(nIter):
    output = F.conv_transpose2d(input, weight, stride=2, padding=(K-1)//2, output_padding=0)

end = time.time()

time_taken = end - start
flopsPerConvTranspose = 2.0*C*W*W*M*K*K

gigaFlops = (flopsPerConvTranspose * 1.0e-9) * nIter / time_taken

print("Performance=%.2f GFlop/s, Iterations=%d, Total Time=%.3f seconds, Ops per Iteration=%.0f"
     % ( gigaFlops, nIter, time_taken, flopsPerConvTranspose))
