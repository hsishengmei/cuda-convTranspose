from torch import nn
import torch
import torch.nn.functional as F
import time
import sys 

K = 4

layer = int(sys.argv[1])

if layer == 0:
    W = 4
    C = 512
    M = 256
elif layer == 1:
    W = 8
    C = 256
    M = 128
elif layer == 2:
    W = 16
    C = 128
    M = 64
elif layer == 3:
    W = 32
    C = 64
    M = 3

print("layer %d: W=%d C=%d M=%d K=%d" % (layer, W, C, M, K))

if torch.cuda.is_available():
    print("cuda is available")

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
