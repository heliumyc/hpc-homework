//
// Created by CONG YU on 4/20/20.
//

#include <algorithm>
#include <cstdio>
#include <omp.h>
#include <random>
#include <string>

double compare_vec(double *v1, double *v2, long n) {
    double diff = 0;
#pragma omp parallel for reduction (+: diff)
    for (int i = 0; i < n; i++) {
        diff += std::abs(v1[i] - v2[i]);
    }
    return diff;
}

void sequential_vec_mat_mul(double *res, const double *mat, const double *vec, long n) {
    for (long i = 0; i < n; i++) {
        double sum = 0;
        for (int j = 0; j < n; j++) {
            sum += vec[j] * mat[j + i * n];
        }
        res[i] = sum;
    }
}

void openmp_vec_mat_mul(double *res, const double *mat, const double *vec, long n) {
#pragma omp parallel for schedule(static)
    for (long i = 0; i < n; i++) {
        double sum = 0;
#pragma omp parallel for reduction (+: sum)
        for (int j = 0; j < n; j++) {
            sum += vec[j] * mat[j + i * n];
        }
        res[i] = sum;
    }
}

#define TILE_LEN 8 // block size be 8*8=64

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

__global__ void gpu_mat_vec_mul(const double* mat, const double* vec, double* result, int n){

    __shared__ double smem[TILE_LEN][TILE_LEN];

    long idx = threadIdx.x + blockIdx.x*blockDim.x;
    long idy = threadIdx.y + blockIdx.y*blockDim.y;

    if(idx < n && idy < n){
        smem[threadIdx.x][threadIdx.y] = mat[idx*n + idy] * vec[idy]; // mat(idx, idy) * vec (idy)
        __syncthreads();
    }

    for (long s = blockDim.y /2; s>0; s >>=1) {
        if (threadIdx.y < s) {
            smem[threadIdx.x][threadIdx.y] += smem[threadIdx.x][threadIdx.y+s];
        }
        __syncthreads();
    }

    if(threadIdx.y == 0){
        atomicAdd2(result + idx, smem[threadIdx.x][threadIdx.y]);
    }
}

void Check_CUDA_Error(const char *message) {
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        fprintf(stderr, "ERROR: %s: %s\n", message, cudaGetErrorString(error));
        exit(-1);
    }
}

int main() {

    long n = 1 << 13;
    double *vec;
    double *mat;
    double *vec_ref;
    double *vec_mul;
    cudaMallocHost((void **) &vec, n * sizeof(double));
    cudaMallocHost((void **) &mat, n * n * sizeof(double));
    cudaMallocHost((void **) &vec_ref, n * sizeof(double));
    cudaMallocHost((void **) &vec_mul, n * sizeof(double));

    // random
    std::random_device rd;
    std::default_random_engine gen(rd());
    std::uniform_real_distribution<double> uniformRealDistribution(-1, 1);

    for (long i = 0; i < n; i++) {
        vec[i] = uniformRealDistribution(gen);
        for (long j = 0; j < n; j++) {
            mat[j + i * n] = uniformRealDistribution(gen);
        }
    }

    double time;

    // sequential calculation
    double tick;

    tick = omp_get_wtime();
    sequential_vec_mat_mul(vec_ref, mat, vec, n);
    time = omp_get_wtime() - tick;
    printf("Sequential benchmark\n");
    printf("Time = %f\n", time);
    printf("CPU Bandwidth = %f GB/s\n", (2*n * n + n) * sizeof(double) / time / 1e9);
    printf("Error = %f\n", compare_vec(vec_ref, vec_ref, n));

    printf("------------\n");

    // openmp calculation
    tick = omp_get_wtime();
    openmp_vec_mat_mul(vec_mul, mat, vec, n);
    time = omp_get_wtime() - tick;
    printf("Openmp benchmark\n");
    printf("Time = %f\n", time);
    printf("CPU Bandwidth = %f GB/s\n", (2*n * n + n) * sizeof(double) / time / 1e9);
    printf("Error = %f\n", compare_vec(vec_ref, vec_mul, n));

    printf("------------\n");

    // cuda
    std::fill(vec_mul, vec_mul+n, 0);

    // init
    double *vec_d;
    double *mat_d;
    double *gpu_result;
    cudaMalloc(&vec_d, n * sizeof(double));
    Check_CUDA_Error("malloc vec_d failed");
    cudaMalloc(&mat_d, n * n * sizeof(double));
    Check_CUDA_Error("malloc mat_d failed");
    cudaMalloc(&gpu_result, n * sizeof(double));
    Check_CUDA_Error("malloc temp_mat failed");
    cudaMemcpyAsync(vec_d, vec, n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(gpu_result, vec_mul, n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(mat_d, mat, n * n * sizeof(double), cudaMemcpyHostToDevice);

    dim3 grid(n/TILE_LEN, n/TILE_LEN);
    dim3 block(TILE_LEN, TILE_LEN);
    cudaDeviceSynchronize();

    tick = omp_get_wtime();
    gpu_mat_vec_mul<<< grid,block >>>(mat_d, vec_d, gpu_result, n);
    cudaDeviceSynchronize();
    Check_CUDA_Error("mul failed");
    // fetch mul result
    // currently sum_d is out anwser
    cudaMemcpyAsync(vec_mul, gpu_result, n * sizeof(double), cudaMemcpyDeviceToHost);
    Check_CUDA_Error("copy result back failed");
    cudaDeviceSynchronize();

    time = omp_get_wtime() - tick;
    printf("GPU benchmark\n");
    printf("Time = %f\n", time);
    printf("GPU Bandwidth = %f GB/s\n", (2*n*n+n) * sizeof(double) / time / 1e9);
    printf("Error = %f\n", compare_vec(vec_ref, vec_mul, n));

    cudaFreeHost(vec);
    cudaFreeHost(mat);
    cudaFreeHost(vec_ref);
    cudaFreeHost(vec_mul);

    cudaFree(vec_d);
    cudaFree(mat_d);
    cudaFree(gpu_result);

}

