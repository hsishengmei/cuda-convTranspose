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
void gpu_deconv(float *, float *, float *, FmapShape, FilterShape, FmapShape);
void gpu_deconv_tile(float *, float *, float *, FmapShape, FilterShape, FmapShape);
void print_fmap(float *, FmapShape);
__global__ void deconv_kernel(float *, float *, float *, FmapShape, FilterShape, FmapShape);
__global__ void deconv_kernel_tile(float *, float *, float *, FmapShape, FilterShape, FmapShape);

/*****************************************************************/


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
    
    // to measure time taken by a specific part of the code 
    double time_taken;
    clock_t start, end;
    
    
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
    // Edge elements    initialization
    
    for(int i = 0; i < C*W*W; i++)
         ifmap[i] = i*0.00001;
    for(int i = 0; i < C*M*K*K; i++)
        filter[i] = i*0.00001;
    // for(int i = 0; i < 4*M*W*W; i++)
    //     ofmap_cpu[i] = 0.0;
    // for(int i = 0; i < 4*M*W*W; i++)
    //     ofmap_gpu[i] = 0.0;


    printf("CPU start\n");
    start = clock();
    cpu_deconv(ifmap, filter, ofmap_cpu, ifmap_shape, filter_shape, ofmap_shape);
    end = clock();
    printf("CPU end\n");
    time_taken = ((double)(end - start))/ CLOCKS_PER_SEC;
    printf("Time taken for CPU is %lf\n", time_taken);
    
    printf("GPU start\n");
    start = clock();
    // gpu_deconv(ifmap, filter, ofmap_gpu, ifmap_shape, filter_shape, ofmap_shape);
    gpu_deconv_tile(ifmap, filter, ofmap_gpu, ifmap_shape, filter_shape, ofmap_shape);
    end = clock();
    printf("GPU end\n");
    time_taken = ((double)(end - start))/ CLOCKS_PER_SEC;
    printf("Time taken for GPU is %lf\n", time_taken);

    double diff = 0, sum = 0;
    for (int i=0; i<4*M*W*W; i++) {
        sum += ofmap_cpu[i];
        diff += (ofmap_gpu[i] - ofmap_cpu[i]);
    }
    printf("sum: %f\n", sum);
    printf("diff: %f\n", abs(diff));
    printf("error ratio: %f\n", abs(diff) / sum);

	// print_fmap(ofmap_cpu, ofmap_shape);
	// print_fmap(ofmap_gpu, ofmap_shape);

    free(ifmap);
    free(filter);
    free(ofmap_cpu);
    free(ofmap_gpu);
    
    return 0;

}

void cpu_deconv(float * ifmap, float * filter, float * ofmap, 
                                FmapShape ifmap_shape, FilterShape filter_shape, FmapShape ofmap_shape)
{
    // assuming stride is always 2, batch size is always 1
    unsigned int pad = (filter_shape.K - 1) / 2;
    for (int c=0; c<ifmap_shape.C; c++){
        for (int w1=0; w1<ifmap_shape.W; w1++){
            for (int w2=0; w2<ifmap_shape.W; w2++){
                for (int m=0; m<filter_shape.M; m++){
                    for (int k1=0; k1<filter_shape.K; k1++){
                        for (int k2=0; k2<filter_shape.K; k2++){
                            int output_y = w1*2+k1-pad;
                            int output_x = w2*2+k2-pad;
                            if (output_x >= 0 && output_y >= 0 && output_x < ofmap_shape.W && output_y < ofmap_shape.W){
                                unsigned int ifmap_index = get_fmap_index(c, w1, w2, ifmap_shape);
                                unsigned int filter_index = get_filter_index(c, m, k1, k2, filter_shape);
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

__global__ void deconv_kernel(float * ifmap, float * filter, float * ofmap, 
                                                            FmapShape ifmap_shape, FilterShape filter_shape, FmapShape ofmap_shape)
{
    // assuming stride is always 2
    unsigned int thread_index = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int c = thread_index / (ifmap_shape.W * ifmap_shape.W);
    unsigned int w1 = thread_index % (ifmap_shape.W * ifmap_shape.W) / ifmap_shape.W;
    unsigned int w2 = thread_index % ifmap_shape.W;
    unsigned int pad = (filter_shape.K - 1) / 2;
    for (int m=0; m<filter_shape.M; m++){
        for (int k1=0; k1<filter_shape.K; k1++){
            for (int k2=0; k2<filter_shape.K; k2++){
                int output_y = w1*2+k1-pad;
                int output_x = w2*2+k2-pad;
                if (output_x >= 0 && output_y >= 0 && output_x < ofmap_shape.W && output_y < ofmap_shape.W){
                    unsigned int ifmap_index = get_fmap_index(c, w1, w2, ifmap_shape);
                    unsigned int filter_index = get_filter_index(c, m, k1, k2, filter_shape);
                    unsigned int ofmap_index = get_fmap_index(m, output_y, output_x, ofmap_shape);
                    //ofmap[ofmap_index] += ifmap[ifmap_index] * filter[filter_index];
                    atomicAdd(&ofmap[ofmap_index], ifmap[ifmap_index] * filter[filter_index]);
                }
            }
        } 
    } 
}

void gpu_deconv(float * ifmap, float * filter, float * ofmap, 
    FmapShape ifmap_shape, FilterShape filter_shape, FmapShape ofmap_shape)
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

    unsigned int num_threads = ifmap_shape.C*ifmap_shape.W*ifmap_shape.W;
    unsigned int max_num_threads = 1024;
    if (num_threads > max_num_threads)
        deconv_kernel<<<ceil(num_threads / max_num_threads), max_num_threads>>>(d_ifmap, d_filter, d_ofmap, ifmap_shape, filter_shape, ofmap_shape);
    else
        deconv_kernel<<<1, num_threads>>>(d_ifmap, d_filter, d_ofmap, ifmap_shape, filter_shape, ofmap_shape);

    //deconv_kernel<<<(ifmap_shape.W * ifmap_shape.W + num_threads - 1)/num_threads, num_threads>>>(N, d_playground, d_buffer);
    // deconv_kernel<<<1, num_threads>>>(d_ifmap, d_filter, d_ofmap, ifmap_shape, filter_shape, ofmap_shape);
    // deconv_kernel<<<ceil(num_threads / max_num_threads), max_num_threads>>>(d_ifmap, d_filter, d_ofmap, ifmap_shape, filter_shape, ofmap_shape);
    gpuErrchk(cudaMemcpy(ofmap, d_ofmap, ofmap_size, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaFree(d_ifmap));
    gpuErrchk(cudaFree(d_filter));
    gpuErrchk(cudaFree(d_ofmap));
}

template<int BLOCK_SIZE>
__global__ 
void deconv_kernel_tile(float * ifmap, float * filter, float * ofmap, 
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

	// for (int w0=w0_start; w0<w0_end; w0+=BLOCK_SIZE){
	// 	for (int w1=w1_start; w1<w1_end; w1+=BLOCK_SIZE){
			// load input
			__shared__ float In[BLOCK_SIZE][BLOCK_SIZE];
			In[t0][t1] = ifmap[get_fmap_index(c, w0, w1, ifmap_shape)];
			// __syncthreads();

			for (int m=0; m<filter_shape.M; ++m) {
				for (int k0=0; k0<filter_shape.K; ++k0) {
					for (int k1=0; k1<filter_shape.K; ++k1) {

						int output_y = w0*2+k0-pad;
						int output_x = w1*2+k1-pad;
						if (output_x >= 0 && output_y >= 0 && output_x < ofmap_shape.W && output_y < ofmap_shape.W){
							unsigned int filter_index = get_filter_index(c, m, k0, k1, filter_shape);
							unsigned int ofmap_index = get_fmap_index(m, output_y, output_x, ofmap_shape);
							
							//ofmap[ofmap_index] += ifmap[ifmap_index] * filter[filter_index];
							atomicAdd(&ofmap[ofmap_index], In[t0][t1] * filter[filter_index]);
						}
					}
				}
			}
			
			// __syncthreads();
	// 	}
	// }
	
}

void gpu_deconv_tile(float * ifmap, float * filter, float * ofmap, 
    				 FmapShape ifmap_shape, FilterShape filter_shape, FmapShape ofmap_shape)
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

	const unsigned int block_size = 4;
	dim3 threads(block_size, block_size);
	dim3 grid(ifmap_shape.C, ifmap_shape.W / threads.x, ifmap_shape.W / threads.y);

	deconv_kernel_tile<block_size> <<< grid, threads >>> (d_ifmap, d_filter, d_ofmap, 
											ifmap_shape, filter_shape, ofmap_shape);

    //deconv_kernel<<<(ifmap_shape.W * ifmap_shape.W + num_threads - 1)/num_threads, num_threads>>>(N, d_playground, d_buffer);
    // deconv_kernel<<<1, num_threads>>>(d_ifmap, d_filter, d_ofmap, ifmap_shape, filter_shape, ofmap_shape);
    // deconv_kernel<<<ceil(num_threads / max_num_threads), max_num_threads>>>(d_ifmap, d_filter, d_ofmap, ifmap_shape, filter_shape, ofmap_shape);
    gpuErrchk(cudaMemcpy(ofmap, d_ofmap, ofmap_size, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaFree(d_ifmap));
    gpuErrchk(cudaFree(d_filter));
    gpuErrchk(cudaFree(d_ofmap));
}


void print_fmap(float * fmap, FmapShape fmap_shape){
    printf("%d %d %d\n", fmap_shape.C, fmap_shape.W, fmap_shape.W);
    for (int c = 0; c < fmap_shape.C; c++) {
        for (int w1 = 0; w1 < fmap_shape.W; w1++) {
            for (int w2 = 0; w2 < fmap_shape.W; w2++) {
                unsigned int fmap_index = get_fmap_index(c, w1, w2, fmap_shape);
                printf("%f ", fmap[fmap_index]);
            }
            printf("\n");
        }
        printf("\n");
    }
}
