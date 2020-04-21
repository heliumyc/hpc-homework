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

#define BLOCK_SIZE 32

__global__ void MatrixMulKernel(double* mat, double* vec, double* result, int n){

    __shared__ double block[BLOCK_SIZE][BLOCK_SIZE];  // Shared memory
    __shared__ double slice[BLOCK_SIZE];

    int bx = blockIdx.x;
    int by = blockIdx.y; // ID thread
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * BLOCK_SIZE + ty;
    int col = bx * BLOCK_SIZE + tx;

    double sum = 0;

    // Loop over the mat and Nd tiles required to compute the Pd element
    for (int i = 0; i < n / BLOCK_SIZE; ++i) {
        // Collaborative loading of mat and Nd tiles into shared memory
        Mds[ty][tx] = mat[row * n + (i * BLOCK_SIZE + tx)]; // mat(row, i*tile+tx)
        Nds[ty][tx] = vec[col + (i * BLOCK_SIZE + ty) * n]; // vec(col + (i*tile+ty)*n)

        __syncthreads();

        for (int k = 0; k < BLOCK_SIZE; ++k)
            sum +=  Mds[ty][k] * Nds[k][tx];

        __syncthreads();
    }

    result[row * n + col] = sum;
}

void Check_CUDA_Error(const char *message) {
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        fprintf(stderr, "ERROR: %s: %s\n", message, cudaGetErrorString(error));
        exit(-1);
    }
}

int main() {

    long n = 1 << 12;
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
    printf("CPU Bandwidth = %f GB/s\n", (n * n + n) * sizeof(double) / time / 1e9);
    printf("Error = %f\n", compare_vec(vec_ref, vec_ref, n));

    printf("------------\n");

    // openmp calculation
    tick = omp_get_wtime();
    openmp_vec_mat_mul(vec_mul, mat, vec, n);
    time = omp_get_wtime() - tick;
    printf("Openmp benchmark\n");
    printf("Time = %f\n", time);
    printf("CPU Bandwidth = %f GB/s\n", (n * n + n) * sizeof(double) / time / 1e9);
    printf("Error = %f\n", compare_vec(vec_ref, vec_mul, n));

    printf("------------\n");

    // cuda

    // init
    double *vec_d;
    double *mat_d;
    double *temp_mat_d;
    cudaMalloc(&vec_d, n * sizeof(double));
    Check_CUDA_Error("malloc vec_d failed");
    cudaMalloc(&mat_d, n * n * sizeof(double));
    Check_CUDA_Error("malloc mat_d failed");
    cudaMalloc(&temp_mat_d, n * n * sizeof(double));
    Check_CUDA_Error("malloc temp_mat failed");
    cudaMemcpyAsync(vec_d, vec, n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(mat_d, mat, n * n * sizeof(double), cudaMemcpyHostToDevice);


    // fetch mul result
    // currently sum_d is out anwser
    cudaMemcpy(vec_mul, sum_d, n * sizeof(double), cudaMemcpyDeviceToHost);
    Check_CUDA_Error("copy result back failed");
    cudaDeviceSynchronize();

    time = omp_get_wtime() - tick;
    printf("GPU benchmark\n");
    printf("Time = %f\n", time);
    printf("GPU Bandwidth = %f GB/s\n", (n*n+n) * sizeof(double) / time / 1e9);
    printf("Error = %f\n", compare_vec(vec_ref, vec_mul, n));

    cudaFreeHost(vec);
    cudaFreeHost(mat);
    cudaFreeHost(vec_ref);
    cudaFreeHost(vec_mul);

    cudaFree(vec_d);
    cudaFree(mat_d);
    cudaFree(temp_mat_d);
    cudaFree(extra_d);

}

