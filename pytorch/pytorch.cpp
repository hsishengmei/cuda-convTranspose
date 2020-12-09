#include <torch/torch.h>
#include <iostream>
#include <ctime>

int main(int argc, char** argv) {
    int C = 128, W = 16, M = 64, K = 4;
    auto tensorOptions = torch::TensorOptions()
                                .dtype(torch::kFloat32)
                                .layout(torch::kStrided)
                                .requires_grad(false);

    torch::Tensor ifmap = torch::rand({1, C, W, W}, tensorOptions);
    int pad = (K-1) / 2;
    int output_pad = K % 2;
    int stride = 2;

    int nIter = 1000;

    torch::Tensor ofmap;
    torch::nn::ConvTranspose2d deconv(torch::nn::ConvTranspose2dOptions(C, M, K)
                                                .stride(stride)
                                                .padding(pad)
                                                .output_padding(output_pad)
                                                .bias(false));  

    printf("start\n");
    std::time_t start = std::time(nullptr);

    for (int i=0; i<nIter; ++i) {
        ofmap = deconv(ifmap);
    }
    

    double time_taken = std::difftime(std::time(nullptr), start);
    printf("Time taken for GPU is %lf seconds\n", time_taken);
        
    double flopsPerConvTranspose = 2.0*C*W*W*M*K*K;

    double gigaFlops = (flopsPerConvTranspose * 1.0e-9f) * nIter / time_taken;
    printf(
        "Performance=%.2f GFlop/s, Iterations=%d, Total Time=%.3f seconds, Ops per Iteration=%.0f\n",
        gigaFlops,
        nIter,
        time_taken,
        flopsPerConvTranspose);

}
