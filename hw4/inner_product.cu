//
// Created by CONG YU on 4/20/20.
//
#include <algorithm>
#include <stdio.h>
#include <omp.h>
#include <random>
#include <string>

// cpu sequential computation
void sequential_vec_inner_product(double *res, const double *a, const double *b, long n) {
    double acc = 0;
    for (long i = 0; i < n; i++) {
        acc += a[i]*b[i];
    }
    *res = acc;
}

// openmp cpu map
void openmp_inner_product(double *res, const double *a, const double *b, long n) {
    double sum = 0;
#pragma omp parallel for reduction (+: sum)
    for (long i = 0; i < n; i++) {
        sum += a[i] * b[i];
    }
    *res = sum;
}

#define BLOCK_SIZE 1024

// gpu reduce
__global__
void gpu_map_vec_inner_product(const double*a, const double *b, double *c, long n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] * b[idx];
    }
}

// gpu reduce
__global__ void gpu_reduce_inner_product(double* sum, const double* a, long N){
    __shared__ double smem[BLOCK_SIZE];
    int idx = (blockIdx.x) * blockDim.x + threadIdx.x;

    // each thread reads data from global into shared memory
    if (idx < N) smem[threadIdx.x] = a[idx];
    else smem[threadIdx.x] = 0;
    __syncthreads();

    // x >>= 1 means "set x to itself shifted by one bit to the right", i.e., a divison by 2
    // write to memory with threadIdx rather than ``index''
    for (unsigned int s=blockDim.x/2; s>0; s>>=1) {
        if (threadIdx.x < s) {
            smem[threadIdx.x] += smem[threadIdx.x + s];
        }
        __syncthreads();
    }

    // write to global memory
    if (threadIdx.x == 0) sum[blockIdx.x] = smem[threadIdx.x];
}

int main() {
    long n = (1UL<<25); // 2^25

    // malloc
//    auto* a = (double*) malloc(n * sizeof(double));
//    auto* b = (double*) malloc(n * sizeof(double));
//    auto* temp = (double*) malloc(n * sizeof(double));
    double* a;
    double* b;
    double* temp;
    cudaMallocHost((void**)&a, n * sizeof(double));
    cudaMallocHost((void**)&b, n * sizeof(double));
    cudaMallocHost((void**)&temp, n * sizeof(double));

    // random
    std::random_device rd;
    std::default_random_engine gen(rd());
    std::uniform_real_distribution<double> uniformRealDistribution(-1, 1);

    // init
    omp_set_num_threads(6);
#pragma omp parallel for schedule(static)
    for (long i = 0; i< n; i++) {
        a[i] = uniformRealDistribution(gen);
        b[i] = uniformRealDistribution(gen);
    }

    double time;

    // sequential calculation
    double tick;

    tick = omp_get_wtime();
    double ref;
    sequential_vec_inner_product(&ref, a, b, n);
    time = omp_get_wtime() - tick;
    printf("Sequential benchmark\n");
    printf("Time = %f\n", time);
    printf("CPU Bandwidth = %f GB/s\n", 2*n*sizeof(double) / time/1e9);
    printf("Error = %f\n", std::abs(ref-ref));

    printf("------------\n");

    // openmp calculation
    tick = omp_get_wtime();
    double openmp_res;
    openmp_inner_product(&openmp_res, a, b, n);
    time = omp_get_wtime() - tick;
    printf("Openmp benchmark\n");
    printf("Time = %f\n", time);
    printf("CPU Bandwidth = %f GB/s\n", 2*n*sizeof(double) / time/1e9);
    printf("Error = %f\n", std::abs(openmp_res-ref));

    printf("------------\n");

    // cuda

    // init
    double* a_d;
    double* b_d;
    cudaMalloc(&a_d, n*sizeof(double));
    cudaMalloc(&b_d, n*sizeof(double));
    cudaMemcpyAsync(a_d, a, n*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(b_d, b, n*sizeof(double), cudaMemcpyHostToDevice);

    double* temp_d;
    cudaMalloc(&temp_d, n*sizeof(double));
    double* extra_d;
    long N_work = 1;
    for (long i = (n+BLOCK_SIZE-1)/(BLOCK_SIZE); i > 1; i = (i+BLOCK_SIZE-1)/(BLOCK_SIZE)) N_work += i;
    cudaMalloc(&extra_d, N_work*sizeof(double)); // extra memory buffer for reduction across thread-blocks
    cudaDeviceSynchronize();

    tick = omp_get_wtime();
    double cuda_res;

    gpu_map_vec_inner_product<<<n/BLOCK_SIZE,BLOCK_SIZE>>>(a_d, b_d, temp_d, n);

    double* sum_d = extra_d;
    long Nb = (n+BLOCK_SIZE-1)/(BLOCK_SIZE);
    gpu_reduce_inner_product << < Nb, BLOCK_SIZE >> > (sum_d, temp_d, n);
    while (Nb > 1) {
        long lastN = Nb;
        Nb = (Nb+BLOCK_SIZE-1)/(BLOCK_SIZE);
        gpu_reduce_inner_product << < Nb, BLOCK_SIZE >> > (sum_d + lastN, sum_d, lastN);
        sum_d += lastN;
    }
    cudaMemcpyAsync(&cuda_res, sum_d, 1*sizeof(double), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    time = omp_get_wtime() - tick;
    printf("GPU benchmark\n");
    printf("Time = %f\n", time);
    printf("GPU Bandwidth = %f GB/s\n", 1*n*sizeof(double) / time/1e9);
    printf("Error = %f\n", std::abs(cuda_res-ref));

    // free
    cudaFreeHost(a);
    cudaFreeHost(b);
    cudaFreeHost(temp);
    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(temp_d);
    cudaFree(extra_d);
}
