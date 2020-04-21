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

__device__ double global_sum;

__device__ double atomicAdd2(double* address, double val)
{
    auto* address_as_ull =
            (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val +
                                             __longlong_as_double(assumed)));

        // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);

    return __longlong_as_double(old);
}

// gpu reduce
__global__ void gpu_inner_product(const double *a, const double *b, long N) {
    __shared__ double smem[BLOCK_SIZE];
    long idx = blockIdx.x*blockDim.x + threadIdx.x; // idx on one dim vector
    if (idx < N) {
        smem[threadIdx.x] = a[idx] * b[idx];
    }
    else smem[threadIdx.x] = 0;

    __syncthreads();

    for (unsigned int s=blockDim.x/2; s>0; s>>=1) {
        if (threadIdx.x < s) {
            smem[threadIdx.x] += smem[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) atomicAdd2(&global_sum, smem[0]);
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
    double cuda_res;
    cudaMalloc(&a_d, n*sizeof(double));
    cudaMalloc(&b_d, n*sizeof(double));
    cudaMemcpyAsync(a_d, a, n*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(b_d, b, n*sizeof(double), cudaMemcpyHostToDevice);
    /// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    /// this is essential!!!!!!!!! must move to memory of GPU, else it will be in main mem and ultra SLOW
    /// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    cudaMemcpyToSymbol(global_sum, &cuda_res, sizeof(double));
    cudaDeviceSynchronize();

    tick = omp_get_wtime();
    gpu_inner_product<<<n/BLOCK_SIZE,BLOCK_SIZE>>>(a_d, b_d, n);
    cudaMemcpyFromSymbol(&cuda_res, global_sum, sizeof(double));
    cudaDeviceSynchronize();

    time = omp_get_wtime() - tick;
    printf("GPU benchmark\n");
    printf("Time = %f\n", time);
    printf("GPU Bandwidth = %f GB/s\n", 2*n*sizeof(double) / time/1e9);
    printf("Error = %f\n", std::abs(cuda_res-ref));

    // free
    cudaFreeHost(a);
    cudaFreeHost(b);
    cudaFreeHost(temp);
    cudaFree(a_d);
    cudaFree(b_d);
}
