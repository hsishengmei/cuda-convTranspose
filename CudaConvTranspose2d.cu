#include <cuda.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h> 

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
	if (code != cudaSuccess) 
	{
		fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}
/*****************************************************************/
typedef struct FmapShape
{
    unsigned int C;
    unsigned int W;
} FmapShape;

typedef struct FilterShape
{
    unsigned int C;
    unsigned int M;
    unsigned int K;
} FilterShape;

// Function declarations: Feel free to add any functions you want.
__host__ __device__ unsigned int get_fmap_index(unsigned int d1, unsigned int d2, unsigned int d3, FmapShape fmap_shape){
    // Input shape: C*W*W
    return d1 * fmap_shape.W * fmap_shape.W + d2 * fmap_shape.W + d3;
}

__host__ __device__ unsigned int get_filter_index(unsigned int d1, unsigned int d2, unsigned int d3, unsigned int d4, FilterShape filter_shape){
    // Filter shape: C*M*K*K
    return d1 * filter_shape.M * filter_shape.K * filter_shape.K + d2 * filter_shape.K * filter_shape.K + d3 * filter_shape.K + d4;
}
void cpu_deconv(float *, float *, float *, FmapShape, FilterShape, FmapShape);
void gpu_deconv(float *, float *, float *, FmapShape, FilterShape, FmapShape, unsigned int, unsigned int);
void calc_flops(double, unsigned int, unsigned int);
void print_fmap(float *, FmapShape);
__global__ void deconv_kernel(float *, float *, float *, FmapShape, FilterShape, FmapShape);
template<int,int,int> __global__ void deconv_kernel_share_filter_tiled(float *, float *, float *, FmapShape, FilterShape, FmapShape, int);

/*****************************************************************/

// to measure time taken by a specific part of the code 
double seconds;
clock_t start, end;

int main(int argc, char * argv[])
{
    // layer parameters
    unsigned int W, C, M, K;
    K = 4;    // kernel size fixed: 4x4

    // decides which algorithm to use
    int option = atoi(argv[1]);

    // decides layer parameters
    int layer = atoi(argv[2]);
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


    FmapShape ifmap_shape;
    ifmap_shape.C = C;
    ifmap_shape.W = W;
    FilterShape filter_shape;
    filter_shape.C = C;
    filter_shape.M = M;
    filter_shape.K = K;
    FmapShape ofmap_shape;
    ofmap_shape.C = M;
    ofmap_shape.W = 2*W;
    
    /* The 3D array of points will be treated as 1D array of CxWxW elements */
    float * ifmap, * filter, * ofmap_cpu, * ofmap_gpu; 
    
    /* Dynamically allocate NxN array of floats */
    ifmap = (float *)calloc(C*W*W, sizeof(float));
    filter = (float *)calloc(C*M*K*K, sizeof(float));
    ofmap_cpu = (float *)calloc(4*M*W*W, sizeof(float));
    ofmap_gpu = (float *)calloc(4*M*W*W, sizeof(float));

    printf("ifmap size: %d\n", C*W*W);
    printf("filter size: %d\n", C*M*K*K);
    printf("ofmap size: %d\n", 4*M*W*W);
    // return 0;
    /* Initialize it: calloc already initalized everything to 0 */
    
    for(int i = 0; i < C*W*W; i++)
         ifmap[i] = i*0.00001;
    for(int i = 0; i < C*M*K*K; i++)
        filter[i] = i*0.00001;
    // for(int i = 0; i < 4*M*W*W; i++)
    //     ofmap_cpu[i] = 0.0;
    // for(int i = 0; i < 4*M*W*W; i++)
    //     ofmap_gpu[i] = 0.0;

    int nIter = 1000;
    // int nIter = 1;
    if (option == 0)
        cpu_deconv(ifmap, filter, ofmap_cpu, ifmap_shape, filter_shape, ofmap_shape);
    else
        gpu_deconv(ifmap, filter, ofmap_gpu, ifmap_shape, filter_shape, ofmap_shape, nIter, option);

    // check correctness
    if (nIter == 1) {
        cpu_deconv(ifmap, filter, ofmap_cpu, ifmap_shape, filter_shape, ofmap_shape);
        
        double diff = 0, sum = 0;
        for (int i=0; i<4*M*W*W; i++) {
            sum += ofmap_cpu[i];
            diff += (ofmap_gpu[i] - ofmap_cpu[i]);
        }
        printf("sum: %f\n", sum);
        printf("diff: %f\n", abs(diff));
        printf("diff to sum ratio: %f\n", abs(diff) / sum);
    
        // print_fmap(ofmap_cpu, ofmap_shape);
        // print_fmap(ofmap_gpu, ofmap_shape);
    }

    free(ifmap);
    free(filter);
    free(ofmap_cpu);
    free(ofmap_gpu);
    
    return 0;

}

void cpu_deconv(float * ifmap, float * filter, float * ofmap, 
                FmapShape ifmap_shape, FilterShape filter_shape, FmapShape ofmap_shape)
{
    start = clock();

    unsigned int pad = (filter_shape.K - 1) / 2;
    int nIter = 10;
    for (int i=0; i<nIter; ++i) {
        // assuming stride is always 2, batch size is always 1
        for (int c=0; c<ifmap_shape.C; c++){
            for (int w0=0; w0<ifmap_shape.W; w0++){
                for (int w1=0; w1<ifmap_shape.W; w1++){
                    for (int m=0; m<filter_shape.M; m++){
                        for (int k0=0; k0<filter_shape.K; k0++){
                            for (int k1=0; k1<filter_shape.K; k1++){
                                int output_y = w0*2+k0-pad;
                                int output_x = w1*2+k1-pad;
                                if (output_x >= 0 && output_y >= 0 && output_x < ofmap_shape.W && output_y < ofmap_shape.W){
                                    unsigned int ifmap_index = get_fmap_index(c, w0, w1, ifmap_shape);
                                    unsigned int filter_index = get_filter_index(c, m, k0, k1, filter_shape);
                                    unsigned int ofmap_index = get_fmap_index(m, output_y, output_x, ofmap_shape);
                                    ofmap[ofmap_index] += ifmap[ifmap_index] * filter[filter_index];
                                }
                            }
                        } 
                    } 
                }
            }
        }
    }

    end = clock();
    seconds = ((double)(end - start))/ CLOCKS_PER_SEC;
    unsigned int opsPerConvTranspose = 2 * ifmap_shape.C *ifmap_shape.W * ifmap_shape.W *
                                            filter_shape.M * filter_shape.K * filter_shape.K;

    calc_flops(seconds, opsPerConvTranspose, nIter);
}

__global__ 
void deconv_kernel(float * ifmap, float * filter, float * ofmap, 
                    FmapShape ifmap_shape, FilterShape filter_shape, FmapShape ofmap_shape)
{

    // assuming stride is always 2
    unsigned int thread_index = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int c = thread_index / (ifmap_shape.W * ifmap_shape.W);
    unsigned int w0 = thread_index % (ifmap_shape.W * ifmap_shape.W) / ifmap_shape.W;
    unsigned int w1 = thread_index % ifmap_shape.W;
    unsigned int pad = (filter_shape.K - 1) / 2;
    for (int m=0; m<filter_shape.M; m++){
        for (int k0=0; k0<filter_shape.K; k0++){
            for (int k1=0; k1<filter_shape.K; k1++){
                int output_y = w0*2+k0-pad;
                int output_x = w1*2+k1-pad;
                if (output_x >= 0 && output_y >= 0 && output_x < ofmap_shape.W && output_y < ofmap_shape.W){
                    unsigned int ifmap_index = get_fmap_index(c, w0, w1, ifmap_shape);
                    unsigned int filter_index = get_filter_index(c, m, k0, k1, filter_shape);
                    unsigned int ofmap_index = get_fmap_index(m, output_y, output_x, ofmap_shape);
                    //ofmap[ofmap_index] += ifmap[ifmap_index] * filter[filter_index];
                    atomicAdd(&ofmap[ofmap_index], ifmap[ifmap_index] * filter[filter_index]);
                }
            }
        } 
    } 
}

// share filter & input
template<int Tm, int K, int W>
__global__ 
void deconv_kernel_share_filter_tiled(float * ifmap, float * filter, float * ofmap, 
						    FmapShape ifmap_shape, FilterShape filter_shape, FmapShape ofmap_shape, int m_offset)
{
    unsigned int w0 = threadIdx.x % (ifmap_shape.W * ifmap_shape.W) / ifmap_shape.W;
    unsigned int w1 = threadIdx.x % ifmap_shape.W;
    unsigned int pad = (filter_shape.K - 1) / 2;

    // process one input channel per block	
    unsigned int c = blockIdx.x;
    // assume num of threads is a multiple of W*W, each thread should be responsible for multiple m's
    unsigned int m_parallel = blockDim.x / (ifmap_shape.W * ifmap_shape.W);
    unsigned int m_per_thread = (Tm + m_parallel - 1)/ m_parallel;

    __shared__ float filter_shared[Tm][K][K];
    for (int i = threadIdx.x; i < Tm*K*K; i += blockDim.x) {
        unsigned int filter_m = i / (filter_shape.K * filter_shape.K);
        unsigned int filter_k0 = i % (filter_shape.K * filter_shape.K) / filter_shape.K;
        unsigned int filter_k1 = i % filter_shape.K;
        filter_shared[filter_m][filter_k0][filter_k1] = filter[get_filter_index(c, m_offset+filter_m, filter_k0, filter_k1, filter_shape)];
    }

    __shared__ float ofmap_shared[Tm][2*W][2*W];
    for (int i = threadIdx.x; i < Tm*2*W*2*W; i += blockDim.x) {
        unsigned int ofmap_m = i / (ofmap_shape.W * ofmap_shape.W);
        unsigned int ofmap_w0 = i % (ofmap_shape.W * ofmap_shape.W) / ofmap_shape.W;
        unsigned int ofmap_w1 = i % ofmap_shape.W;
        ofmap_shared[ofmap_m][ofmap_w0][ofmap_w1] = 0;
    }

    __syncthreads();
    unsigned int m = threadIdx.x / (ifmap_shape.W * ifmap_shape.W);
    unsigned int ifmap_index = get_fmap_index(c, w0, w1, ifmap_shape);

    // divide the output window per filter into 4 blocks to avoid memory access conflicts
    for (int b0=0; b0<2; b0++){
        for (int b1=0; b1<2; b1++){
            for (int mm=m_per_thread*m; mm<m_per_thread*(m+1); mm++){
                for (int k0=b0*2; k0<(b0+1)*2; k0++){
                    for (int k1=b1*2; k1<(b1+1)*2; k1++){
                        int output_y = w0*2+k0-pad;
                        int output_x = w1*2+k1-pad;
                        if (output_x >= 0 && output_y >= 0 && output_x < ofmap_shape.W && output_y < ofmap_shape.W && mm<Tm){
                            ofmap_shared[mm][output_y][output_x] += ifmap[ifmap_index] * filter_shared[mm][k0][k1];
                        }
                    }
                }
            }
            __syncthreads();
        }
    }

    for (int i = threadIdx.x; i < Tm*2*W*2*W; i += blockDim.x) {
        unsigned int ofmap_m = i / (ofmap_shape.W * ofmap_shape.W);
        unsigned int ofmap_w0 = i % (ofmap_shape.W * ofmap_shape.W) / ofmap_shape.W;
        unsigned int ofmap_w1 = i % ofmap_shape.W;
        atomicAdd(&ofmap[get_fmap_index(m_offset+ofmap_m, ofmap_w0, ofmap_w1, ofmap_shape)], ofmap_shared[ofmap_m][ofmap_w0][ofmap_w1]);
    }
}

void gpu_deconv(float * ifmap, float * filter, float * ofmap, 
                FmapShape ifmap_shape, FilterShape filter_shape, FmapShape ofmap_shape, 
                unsigned int nIter, unsigned int option)
{
    float * d_ifmap, * d_filter, * d_ofmap;
    size_t ifmap_size = ifmap_shape.C*ifmap_shape.W*ifmap_shape.W*sizeof(float);
    size_t filter_size = filter_shape.C*filter_shape.M*filter_shape.K*filter_shape.K*sizeof(float);
    size_t ofmap_size = ofmap_shape.C*ofmap_shape.W*ofmap_shape.W*sizeof(float);

    gpuErrchk(cudaMalloc(&d_ifmap, ifmap_size));
    gpuErrchk(cudaMalloc(&d_filter, filter_size));
    gpuErrchk(cudaMalloc(&d_ofmap, ofmap_size));
   
    gpuErrchk(cudaMemcpy(d_ifmap, ifmap, ifmap_size, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_filter, filter, filter_size, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_ofmap, ofmap, ofmap_size, cudaMemcpyHostToDevice));

    
    start = clock();

    // naive impl
    if (option == 1) {
        int nBlocks = ceil(1.0*ifmap_shape.C*ifmap_shape.W*ifmap_shape.W)/1024;
        for (int j=0; j<nIter; ++j) {
            deconv_kernel <<< nBlocks, 1024 >>> 
                (d_ifmap, d_filter, d_ofmap, ifmap_shape, filter_shape, ofmap_shape);
        }
    }
    // shared filter/ofmap + tiling
    else if (option == 2) {
        unsigned int threads = 1024;
        dim3 grid(filter_shape.C);
        for (int j=0; j<nIter; ++j) {
            if (filter_shape.M == 256) {
                // M = 256, W = 4, Tm = 150, remaining Tm = 106
                const unsigned int W = 4;
                const unsigned int Tm = 150;
                const unsigned int remaining_Tm = 106; // M % Tm
                for (int k=0; k<(filter_shape.M+Tm-1)/Tm-1; ++k) {
                    deconv_kernel_share_filter_tiled<Tm, 4, W> <<< grid, threads >>> 
                        (d_ifmap, d_filter, d_ofmap, ifmap_shape, filter_shape, ofmap_shape, k*Tm);
                }
                deconv_kernel_share_filter_tiled<remaining_Tm, 4, W> <<< grid, threads >>> 
                    (d_ifmap, d_filter, d_ofmap, ifmap_shape, filter_shape, ofmap_shape, ((filter_shape.M+Tm-1)/Tm-1)*Tm);

            }
            else if (filter_shape.M == 128) {
                // M = 128, W = 8, Tm = 45, remaining Tm = 38
                const unsigned int W = 8;
                const unsigned int Tm = 45;
                const unsigned int remaining_Tm = 38; // M % Tm
                for (int k=0; k<(filter_shape.M+Tm-1)/Tm-1; ++k) {
                    deconv_kernel_share_filter_tiled<Tm, 4, W> <<< grid, threads >>> 
                        (d_ifmap, d_filter, d_ofmap, ifmap_shape, filter_shape, ofmap_shape, k*Tm);
                }
                deconv_kernel_share_filter_tiled<remaining_Tm, 4, W> <<< grid, threads >>> 
                    (d_ifmap, d_filter, d_ofmap, ifmap_shape, filter_shape, ofmap_shape, ((filter_shape.M+Tm-1)/Tm-1)*Tm);

            }
            else if (filter_shape.M == 64) {
                // M = 64, W = 16, Tm = 10, remaining Tm = 4
                const unsigned int W = 16;
                const unsigned int Tm = 10;
                const unsigned int remaining_Tm = 4; // M % Tm
                for (int k=0; k<(filter_shape.M+Tm-1)/Tm-1; ++k) {
                    deconv_kernel_share_filter_tiled<Tm, 4, W> <<< grid, threads >>> 
                        (d_ifmap, d_filter, d_ofmap, ifmap_shape, filter_shape, ofmap_shape, k*Tm);
                }
                deconv_kernel_share_filter_tiled<remaining_Tm, 4, W> <<< grid, threads >>> 
                    (d_ifmap, d_filter, d_ofmap, ifmap_shape, filter_shape, ofmap_shape, ((filter_shape.M+Tm-1)/Tm-1)*Tm);

            }
            else if (filter_shape.M == 3) {
                // M = 3, W = 32, Tm = 2, remaining Tm = 1
                const unsigned int W = 32;
                const unsigned int Tm = 2;
                const unsigned int remaining_Tm = 1; // M % Tm
                for (int k=0; k<(filter_shape.M+Tm-1)/Tm-1; ++k) {
                    deconv_kernel_share_filter_tiled<Tm, 4, W> <<< grid, threads >>> 
                        (d_ifmap, d_filter, d_ofmap, ifmap_shape, filter_shape, ofmap_shape, k*Tm);
                }
                deconv_kernel_share_filter_tiled<remaining_Tm, 4, W> <<< grid, threads >>> 
                    (d_ifmap, d_filter, d_ofmap, ifmap_shape, filter_shape, ofmap_shape, ((filter_shape.M+Tm-1)/Tm-1)*Tm);

            }
            else {
                printf("layer not found\n");
                exit(1);
            }
            // for (int k=0; k<(filter_shape.M+Tm-1)/Tm-1; ++k) {
            //     deconv_kernel_share_filter_tiled<Tm, 4, 16> <<< grid, threads >>> 
            //         (d_ifmap, d_filter, d_ofmap, ifmap_shape, filter_shape, ofmap_shape, k*Tm);
            // }
            // deconv_kernel_share_filter_tiled<remaining_Tm, 4, 16> <<< grid, threads >>> 
            //     (d_ifmap, d_filter, d_ofmap, ifmap_shape, filter_shape, ofmap_shape, ((filter_shape.M+Tm-1)/Tm-1)*Tm);

        }
        
    }
    gpuErrchk(cudaDeviceSynchronize());
    
    end = clock();
    
    seconds = ((double)(end - start))/ CLOCKS_PER_SEC;
    
    unsigned int opsPerConvTranspose = 2 * ifmap_shape.C *ifmap_shape.W * ifmap_shape.W *
                                            filter_shape.M * filter_shape.K * filter_shape.K;

    calc_flops(seconds, opsPerConvTranspose, nIter);

    gpuErrchk(cudaMemcpy(ofmap, d_ofmap, ofmap_size, cudaMemcpyDeviceToHost));

    gpuErrchk(cudaFree(d_ifmap));
    gpuErrchk(cudaFree(d_filter));
    gpuErrchk(cudaFree(d_ofmap));
}

void calc_flops(double seconds, unsigned int opsPerConvTranspose, unsigned int nIter) {
    double gigaFlops = 1.0 * (opsPerConvTranspose * 1.0e-9f) * nIter / seconds;
    printf("Iterations=%d, Total Time=%.2f seconds, Ops per Iteration=%d\n", nIter, seconds, opsPerConvTranspose);
    printf("Performance=%.2f GFlop/sec\n", gigaFlops);
    printf("Average time per iteration=%.2f msec\n\n", seconds * 1000 / nIter);
}

void print_fmap(float * fmap, FmapShape fmap_shape) {
    printf("%d %d %d\n", fmap_shape.C, fmap_shape.W, fmap_shape.W);
    for (int c = 0; c < fmap_shape.C; c++) {
        for (int w0 = 0; w0 < fmap_shape.W; w0++) {
            for (int w1 = 0; w1 < fmap_shape.W; w1++) {
                unsigned int fmap_index = get_fmap_index(c, w0, w1, fmap_shape);
                printf("%f ", fmap[fmap_index]);
            }
            printf("\n");
        }
        printf("\n");
    }
}
