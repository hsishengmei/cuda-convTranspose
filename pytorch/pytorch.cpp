#include <torch/torch.h>
#include <iostream>
#include <chrono>

int main(int argc, char** argv) {
    int C, W, M, K;
    K = 4;

    int layer = atoi(argv[1]);
    if (layer == 0) {
        C = 512;
        M = 256;
        W = 4;
    }
    else if (layer == 1) {
        C = 256;
        M = 128;
        W = 8;
    }
    else if (layer == 2) {
        C = 128;
        M = 64;
        W = 16;
    }
    else if (layer == 3) {
        C = 64;
        M = 3;
        W = 32;
    }
    else {
        printf("layer not found\n");
        exit(1);
    }
    printf("layer %d: C=%d M=%d W=%d K=%d\n", layer, C, M, W, K);
    
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

    auto start = std::chrono::high_resolution_clock::now();

    for (int i=0; i<nIter; ++i) {
        ofmap = deconv(ifmap);
    }
    

    auto end = std::chrono::high_resolution_clock::now();    
    std::chrono::duration<double, std::milli> duration = end - start;
    auto ms_taken = duration.count();
        
    double flopsPerConvTranspose = 2.0*C*W*W*M*K*K;

    double gigaFlops = 1.0 * (flopsPerConvTranspose * 1.0e-9f) * nIter / ms_taken * 1000;
    printf(
        "Performance=%.2f GFlop/s, Iterations=%d, Total Time=%.3f seconds, Ops per Iteration=%.0f\n",
        gigaFlops,
        nIter,
        ms_taken/1000,
        flopsPerConvTranspose);

}
