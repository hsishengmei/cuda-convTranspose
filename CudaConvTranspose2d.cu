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
void cpu_deconv_group_by_ofmap(float *, float *, float *, FmapShape, FilterShape, FmapShape);
void gpu_deconv(float *, float *, float *, FmapShape, FilterShape, FmapShape, unsigned int, unsigned int);
void print_fmap(float *, FmapShape);
__global__ void deconv_kernel(float *, float *, float *, FmapShape, FilterShape, FmapShape);
template<int> __global__ void deconv_kernel_tile_ifmap(float *, float *, float *, FmapShape, FilterShape, FmapShape);
template<int,int,int> __global__ void deconv_kernel_share_filter(float *, float *, float *, FmapShape, FilterShape, FmapShape);
__global__ void deconv_kernel_group_by_ofmap(float *, float *, float *, FmapShape, FilterShape, FmapShape);
template<int,int> __global__ void deconv_kernel_group_by_ofmap_share_ifmap(float *, float *, float *, FmapShape, FilterShape, FmapShape);
template<int,int,int> __global__ void deconv_kernel_group_by_ofmap_share_filter(float *, float *, float *, FmapShape, FilterShape, FmapShape);

/*****************************************************************/

// to measure time taken by a specific part of the code 
double time_taken;
clock_t start, end;

int main(int argc, char * argv[])
{
    unsigned int W, C, M, K;
    // int type_of_device = 0; // CPU or GPU
    W = 16; // input size: W*W
    C = 128; // input channel
    M = 64; // output channel
    K = 4;    // kernel size: K*K

    // W = 16;
    // C = 4;
    // M = 4;
	// K = 4;
	
    // W = 4;
    // C = 1;
    // M = 1;
    // K = 4;

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

    int option = atoi(argv[1]);
    int nIter = 1000;
    // nIter = 1;

    gpu_deconv(ifmap, filter, ofmap_gpu, ifmap_shape, filter_shape, ofmap_shape, nIter, option);
    // cpu_deconv_group_by_ofmap(ifmap, filter, ofmap_gpu, ifmap_shape, filter_shape, ofmap_shape);

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
    printf("CPU start\n");
    start = clock();

    // assuming stride is always 2, batch size is always 1
    unsigned int pad = (filter_shape.K - 1) / 2;
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
    
    end = clock();
    printf("CPU end\n");
    time_taken = ((double)(end - start))/ CLOCKS_PER_SEC;
    printf("Time taken for CPU is %lf\n", time_taken);
}


void cpu_deconv_group_by_ofmap(float * ifmap, float * filter, float * ofmap, 
    FmapShape ifmap_shape, FilterShape filter_shape, FmapShape ofmap_shape)
{
    printf("CPU reorder start\n");
    start = clock();

    // assuming stride is always 2, batch size is always 1
    unsigned int pad = (filter_shape.K - 1) / 2;

    for (int m=0; m<ofmap_shape.C; m++) {
        for (int p0=0; p0<ofmap_shape.W; p0++) {
            for (int p1=0; p1<ofmap_shape.W; p1++) {
                float ofmap_local = 0;
                for (int c=0; c<ifmap_shape.C; c++) {
                    for (int w0=0; w0<ifmap_shape.W; w0++) {
                        for (int w1=0; w1<ifmap_shape.W; w1++) {
                            int k0 = p0 - w0*2 + pad;
                            int k1 = p1 - w1*2 + pad;
                            if (k0 >= 0 && k0 < filter_shape.K && k1 >= 0 && k1 < filter_shape.K) {
                                // ofmap[m][p0][p1] += ifmap[c][w0][w1] * filter[c][m][k0][k1];
                                // ofmap[ofmap_index] += ifmap[get_fmap_index(c, w0, w1, ifmap_shape)] \
                                //                         * filter[get_filter_index(c, m, k0, k1, filter_shape)];
                                ofmap_local += ifmap[get_fmap_index(c, w0, w1, ifmap_shape)] \
                                                        * filter[get_filter_index(c, m, k0, k1, filter_shape)];
                            }
                        }
                    }
                }
                int ofmap_index = get_fmap_index(m, p0, p1, ofmap_shape);
                ofmap[ofmap_index] = ofmap_local;
            }
        }
    }
    

    end = clock();
    printf("CPU reorder end\n");
    time_taken = ((double)(end - start))/ CLOCKS_PER_SEC;
    printf("Time taken for CPU reorder is %lf\n", time_taken);
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


template<int BLOCK_SIZE>
__global__ 
void deconv_kernel_tile_ifmap(float * ifmap, float * filter, float * ofmap, 
						    FmapShape ifmap_shape, FilterShape filter_shape, FmapShape ofmap_shape)
{
	// block index
	unsigned int b0 = blockIdx.x;
	unsigned int b1 = blockIdx.y;
	unsigned int b2 = blockIdx.z;
	
	// thread index
	unsigned int t0 = threadIdx.x;
	unsigned int t1 = threadIdx.y;

	// input channel, fixed
	unsigned int c = b0;
	// kernel index
	// unsigned int w0_start = b1*BLOCK_SIZE + t0;
	// unsigned int w0_end = ifmap_shape.W;
	// unsigned int w1_start = b2*BLOCK_SIZE + t1;
	// unsigned int w1_end = ifmap_shape.W;
	unsigned int w0 = b1*BLOCK_SIZE + t0;
	unsigned int w1 = b2*BLOCK_SIZE + t1;

    unsigned int pad = (filter_shape.K - 1) / 2;

    // load ifmap tile
    __shared__ float In[BLOCK_SIZE][BLOCK_SIZE];
    In[t0][t1] = ifmap[get_fmap_index(c, w0, w1, ifmap_shape)];
    // __syncthreads();


    for (int k0=0; k0<filter_shape.K; ++k0) {
        for (int k1=0; k1<filter_shape.K; ++k1) {
            int output_y = w0*2+k0-pad;
            int output_x = w1*2+k1-pad;
            if (output_x >= 0 && output_y >= 0 && output_x < ofmap_shape.W && output_y < ofmap_shape.W) {
                for (int m=0; m<filter_shape.M; ++m) {
                    unsigned int filter_index = get_filter_index(c, m, k0, k1, filter_shape);
                    unsigned int ofmap_index = get_fmap_index(m, output_y, output_x, ofmap_shape);
                    
                    //ofmap[ofmap_index] += ifmap[ifmap_index] * filter[filter_index];
                    atomicAdd(&ofmap[ofmap_index], In[t0][t1] * filter[filter_index]);
                }
            }
        }
    }
	
}

// share filter & input
template<int M, int K, int W>
__global__ 
void deconv_kernel_share_filter(float * ifmap, float * filter, float * ofmap, 
						    FmapShape ifmap_shape, FilterShape filter_shape, FmapShape ofmap_shape)
{
    //unsigned int thread_index = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int thread_index = threadIdx.z * blockDim.y * blockDim.x +  threadIdx.y * blockDim.x  + threadIdx.x;
    unsigned int w0 = thread_index % (ifmap_shape.W * ifmap_shape.W) / ifmap_shape.W;
    unsigned int w1 = thread_index % ifmap_shape.W;
    unsigned int pad = (filter_shape.K - 1) / 2;
	
    // thread index
    unsigned int c = blockIdx.x;
    unsigned int m = threadIdx.x;
    unsigned int k0 = threadIdx.y;
    unsigned int k1 = threadIdx.z;

    __shared__ float filter_shared[M][K][K];
    filter_shared[m][k0][k1] = filter[get_filter_index(c, m, k0, k1, filter_shape)];
    __syncthreads();
    m = thread_index / (ifmap_shape.W * ifmap_shape.W);
    for (int mm=M/4*m; mm<M/4*(m+1); mm++){
    //for (int m; m<M; m++){
        for (int k0=0; k0<filter_shape.K; k0++){
            for (int k1=0; k1<filter_shape.K; k1++){
                int output_y = w0*2+k0-pad;
                int output_x = w1*2+k1-pad;
                if (output_x >= 0 && output_y >= 0 && output_x < ofmap_shape.W && output_y < ofmap_shape.W){
                    unsigned int ifmap_index = get_fmap_index(c, w0, w1, ifmap_shape);
                    //unsigned int filter_index = get_filter_index(c, m, k0, k1, filter_shape);
                    unsigned int ofmap_index = get_fmap_index(mm, output_y, output_x, ofmap_shape);
                    //atomicAdd(&ofmap[ofmap_index], ifmap[ifmap_index] * filter[filter_index]);
                    atomicAdd(&ofmap[ofmap_index], ifmap[ifmap_index] * filter_shared[mm][k0][k1]);
                }
            }
        } 
    }
}


__global__ 
void deconv_kernel_group_by_ofmap(float * ifmap, float * filter, float * ofmap, 
                    FmapShape ifmap_shape, FilterShape filter_shape, FmapShape ofmap_shape)
{
    unsigned int m = blockIdx.x;

    unsigned int p0 = threadIdx.x;
    unsigned int p1 = threadIdx.y;

    // assuming stride is always 2, batch size is always 1
    unsigned int pad = (filter_shape.K - 1) / 2;

    float ofmap_local = 0;
    for (int c=0; c<ifmap_shape.C; c++) {
        for (int w0=0; w0<ifmap_shape.W; w0++) {
            for (int w1=0; w1<ifmap_shape.W; w1++) {
                int k0 = p0 - w0*2 + pad;
                int k1 = p1 - w1*2 + pad;
                if (k0 >= 0 && k0 < filter_shape.K && k1 >= 0 && k1 < filter_shape.K) {
                    // ofmap[m][p0][p1] += ifmap[c][w0][w1] * filter[c][m][k0][k1];
                    // ofmap[ofmap_index] += ifmap[get_fmap_index(c, w0, w1, ifmap_shape)] \
                    //                         * filter[get_filter_index(c, m, k0, k1, filter_shape)];
                    ofmap_local += ifmap[get_fmap_index(c, w0, w1, ifmap_shape)] \
                                            * filter[get_filter_index(c, m, k0, k1, filter_shape)];
                }
            }
        }
    }
    int ofmap_index = get_fmap_index(m, p0, p1, ofmap_shape);
    ofmap[ofmap_index] = ofmap_local;

}

template<int IFMAP_SIZE, int IFMAP_CHANNELS>
__global__ 
void deconv_kernel_group_by_ofmap_share_ifmap(float * ifmap, float * filter, float * ofmap, 
                    FmapShape ifmap_shape, FilterShape filter_shape, FmapShape ofmap_shape)
{
    unsigned int m = blockIdx.x;
    unsigned int p0 = threadIdx.x;
    unsigned int p1 = threadIdx.y;
    unsigned int ofmap_index = get_fmap_index(m, p0, p1, ofmap_shape);
    float ofmap_local_sum = 0;

    // assuming stride is always 2, batch size is always 1
    unsigned int pad = (filter_shape.K - 1) / 2;

    unsigned int threadId = p0 * ofmap_shape.W + p1;

    // __shared__ float ifmap_local[IFMAP_CHANNELS][IFMAP_SIZE][IFMAP_SIZE];
    // in our case, size of this = 128 x 16 x 16 x sizeof(float) = 32768 x 4 bytes = 128KB
    // cuda1: max shared size is 0x2000 = 49152 bytes = 48KB
    // we can do this in 4 iterations
    // ideally the speedup should be 4x of this?

    __shared__ float ifmap_shared[IFMAP_CHANNELS/4][IFMAP_SIZE][IFMAP_SIZE];
    #pragma unroll
    for (int iter=0; iter<4; ++iter) {
        // let IFMAP_CHANNELS threads load ifmap, assume that thread count is greater than channel count 
        // in our case, thread count = 1024, channel count = 128
        if (threadId < IFMAP_CHANNELS/4) {
            for (int i=0; i<IFMAP_SIZE; ++i) 
                for (int j=0; j<IFMAP_SIZE; ++j) 
                ifmap_shared[threadId][i][j] = ifmap[get_fmap_index(threadId + iter*IFMAP_CHANNELS/4, i, j, ifmap_shape)];
        }
        __syncthreads();

        for (int c=0; c<ifmap_shape.C/4; c++) {
            for (int w0=0; w0<ifmap_shape.W; w0++) {
                for (int w1=0; w1<ifmap_shape.W; w1++) {
                    int k0 = p0 - w0*2 + pad;
                    int k1 = p1 - w1*2 + pad;
                    if (k0 >= 0 && k0 < filter_shape.K && k1 >= 0 && k1 < filter_shape.K) {
                        ofmap_local_sum += ifmap_shared[c][w0][w1] \
                                        * filter[get_filter_index(c+iter*IFMAP_CHANNELS/4, m, k0, k1, filter_shape)];
                    }
                }
            }
        }
        ofmap[ofmap_index] = ofmap_local_sum;
    }
}


template<int IFMAP_CHANNELS, int OFMAP_CHANNELS, int FILTER_SIZE>
__global__ 
void deconv_kernel_group_by_ofmap_share_filter(float * ifmap, float * filter, float * ofmap, 
                    FmapShape ifmap_shape, FilterShape filter_shape, FmapShape ofmap_shape)
{
    unsigned int m = blockIdx.x;
    unsigned int p0 = blockIdx.y;
    unsigned int p1 = blockIdx.z;

    // total of 1024 threads
    // unsigned int threadId = threadIdx.z * blockDim.y * blockDim.x + threadIdx.y * blockDim.x + threadIdx.x;
    unsigned int threadId = threadIdx.x;

    // assign sz input every thread
    // in our case, sz = 128*16*16 / 1024 = 32
    int sz = ifmap_shape.C * ifmap_shape.W * ifmap_shape.W / 1024;

    // assuming stride is always 2, batch size is always 1
    unsigned int pad = (filter_shape.K - 1) / 2;


    __shared__ float filter_shared[IFMAP_CHANNELS/16][OFMAP_CHANNELS][FILTER_SIZE][FILTER_SIZE];
    // size = 128 * 64 * 4 * 4 * sizeof(float) = 512KB
    // cuda1: max shared size is 48KB
    // 512 / 48 = 10.xxx
    // use 16 iterations
    // ideally the speedup should be 16x of this?

    // result sums to this
    __shared__ float shared_sum;
    if (threadId == 0) shared_sum = 0;
    
    float local_sum = 0;

    #pragma unroll
    for (int iter=0; iter<16; ++iter) {
        // IFMAP_CHANNELS*OFMAP_CHANNELS/16 = 128*64/16 = 512, thread count = 1024
        if (threadId < IFMAP_CHANNELS*OFMAP_CHANNELS/16) {
            int dim0 = threadId / OFMAP_CHANNELS;
            int dim1 = threadId % OFMAP_CHANNELS;
            #pragma unroll
            for (int i=0; i<FILTER_SIZE; ++i) {
                #pragma unroll
                for (int j=0; j<FILTER_SIZE; ++j) {
                    filter_shared[dim0][dim1][i][j] = filter[get_filter_index(dim0, dim1, i, j, filter_shape)];
                }
            }
        }
        __syncthreads();
        

        for (int i=0; i<sz; ++i) {
            int ifmap_index = threadId * sz + i;
            int c = ifmap_index / (ifmap_shape.W * ifmap_shape.W);
            int w0 = (ifmap_index % (ifmap_shape.W * ifmap_shape.W)) / ifmap_shape.W;
            // int w1 = (ifmap_index % (ifmap_shape.W * ifmap_shape.W)) % ifmap_shape.W;
            int w1 = ifmap_index % ifmap_shape.W;

            int k0 = p0 - w0*2 + pad;
            int k1 = p1 - w1*2 + pad;
            if (k0 >= 0 && k0 < filter_shape.K && k1 >= 0 && k1 < filter_shape.K) {
                local_sum += ifmap[ifmap_index] * filter_shared[c][m][k0][k1];
            }
        }
        
    }
    
    atomicAdd(&shared_sum, local_sum);

    if (threadId == 0) ofmap[get_fmap_index(m, p0, p1, ofmap_shape)] = local_sum;

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
    if (option == 0) {
        int nBlocks = ceil(1.0*ifmap_shape.C*ifmap_shape.W*ifmap_shape.W)/1024;
        for (int j=0; j<nIter; ++j) {
            deconv_kernel <<< nBlocks, 1024 >>> 
                (d_ifmap, d_filter, d_ofmap, ifmap_shape, filter_shape, ofmap_shape);
        }
    }
    // tile ifmap
    else if (option == 1) {
        const unsigned int blockSize = 16;
        dim3 threads(blockSize, blockSize);
        dim3 grid(ifmap_shape.C, ifmap_shape.W / threads.x, ifmap_shape.W / threads.y);
        for (int j=0; j<nIter; ++j) {
            deconv_kernel_tile_ifmap<blockSize> <<< grid, threads >>> 
                (d_ifmap, d_filter, d_ofmap, ifmap_shape, filter_shape, ofmap_shape);
        }
    }
    // tile filter
    else if (option == 2) {
        // const unsigned int blockSize = ifmap_shape.W;
        dim3 threads(filter_shape.M, filter_shape.K, filter_shape.K);
        dim3 grid(filter_shape.C);
        int nBlocks = ceil(1.0*ifmap_shape.C*ifmap_shape.W*ifmap_shape.W)/(filter_shape.M*filter_shape.K*filter_shape.K);
        for (int j=0; j<nIter; ++j) {
            deconv_kernel_share_filter<64,4,16> <<< grid, threads >>> 
                (d_ifmap, d_filter, d_ofmap, ifmap_shape, filter_shape, ofmap_shape);

        }
    }
    // group by ofmap
    //
    else if (option == 3) {
        dim3 threads(ofmap_shape.W, ofmap_shape.W);
        dim3 grid(ofmap_shape.C);
        for (int j=0; j<nIter; ++j) {
            deconv_kernel_group_by_ofmap <<< grid, threads >>> 
                (d_ifmap, d_filter, d_ofmap, ifmap_shape, filter_shape, ofmap_shape);
        }
    }
    // group by ofmap, share ifmap
    // 2x perf than 3
    // feel like too few blocks?z
    else if (option == 4) {
        dim3 threads(ofmap_shape.W, ofmap_shape.W);
        dim3 grid(ofmap_shape.C);
        for (int j=0; j<nIter; ++j) {
            deconv_kernel_group_by_ofmap_share_ifmap<16,128> <<< grid, threads >>> 
                (d_ifmap, d_filter, d_ofmap, ifmap_shape, filter_shape, ofmap_shape);
        }
    }
    // group by ofmap, share filter
    else if (option == 5) {
        // dim3 threads(ifmap_shape.C, ifmap_shape.W, 1024 / (ifmap_shape.C * ifmap_shape.W));  // 1024 threads, deals with the whole input
        dim3 threads(1024);  // 1024 threads, deals with the whole input
        dim3 grid(ofmap_shape.C, ofmap_shape.W, ofmap_shape.W); // 64*32*32 = 65536 blocks, every block counts one output from the sum of all thread's results
        // every block shares all filters if possible 
        for (int j=0; j<nIter; ++j) {
            deconv_kernel_group_by_ofmap_share_filter<128,64,4> <<< grid, threads >>> 
                (d_ifmap, d_filter, d_ofmap, ifmap_shape, filter_shape, ofmap_shape);
        }
    }
    gpuErrchk(cudaDeviceSynchronize());
    
    end = clock();
    time_taken = ((double)(end - start))/ CLOCKS_PER_SEC;
    
    double flopsPerConvTranspose = 2.0 * static_cast<double>(ifmap_shape.C) *
                                        static_cast<double>(ifmap_shape.W) *
                                        static_cast<double>(ifmap_shape.W) *
                                        static_cast<double>(filter_shape.M) *
                                        static_cast<double>(filter_shape.K) *
                                        static_cast<double>(filter_shape.K);

    double gigaFlops = (flopsPerConvTranspose * 1.0e-9f) * nIter / time_taken;
    printf(
        "Performance=%.2f GFlop/s, Iterations=%d, Total Time=%.3f msec, Size=%.0f Ops\n",
        gigaFlops,
        nIter,
        time_taken,
        flopsPerConvTranspose);

    gpuErrchk(cudaMemcpy(ofmap, d_ofmap, ofmap_size, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaFree(d_ifmap));
    gpuErrchk(cudaFree(d_filter));
    gpuErrchk(cudaFree(d_ofmap));
}

void ConstantInit(float* arr, int sz, float val) {
    for (int i=0; i<sz; ++i)
        arr[i] = val;
}

void print_fmap(float * fmap, FmapShape fmap_shape){
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
