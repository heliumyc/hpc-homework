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

__global__
void gpu_map_vec_mat_mul(const double *mat, const double *vec, double *temp_mat, long n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    if (idx < n) {
        temp_mat[idy + idx * n] = mat[idy + idx * n] * vec[idy];
    }
}

// gpu reduce
__global__ void gpu_reduce_vec_mat_mul(double *sum, const double *mat, long n) {
    __shared__ double smem[BLOCK_SIZE][BLOCK_SIZE];
    int idx = (blockIdx.x) * blockDim.x + threadIdx.x;
    int idy = (blockIdx.y) * blockDim.y + threadIdx.y;

    // each thread reads data from global into shared memory
    if (idx < n && idy < n) smem[threadIdx.x][threadIdx.y] = mat[idy + idx * n];
    else smem[threadIdx.x][threadIdx.y] = 0;
    __syncthreads();


    // x >>= 1 means "set x to itself shifted by one bit to the right", i.e., a divison by 2
    // write to memory with threadIdx rather than ``index''
    for (unsigned int s = blockDim.y / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            smem[threadIdx.x][threadIdx.y] += smem[threadIdx.x][threadIdx.y + s];
        }
        __syncthreads();
    }

    // write to global memory
    if (threadIdx.y == 0) {
        int xx = blockIdx.x * blockDim.x + threadIdx.x;
        // new buffer index (blockX*blockX.num + threadOffset,  blockY)
        sum[blockIdx.y * n + xx] = smem[threadIdx.x][threadIdx.y];
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

    long n = 1 << 12;
//    double* vec = (double *) malloc(n * sizeof(double));
//    double* mat = (double *) malloc(n*n * sizeof(double));
//    double* vec_ref = (double *) malloc(n * sizeof(double));
//    double* vec_mul = (double *) malloc(n * sizeof(double));
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

    // alloc extra space 
    double *extra_d;
    long N_work = 1;
    for (long i = (n + BLOCK_SIZE - 1) / (BLOCK_SIZE); i > 1; i = (i + BLOCK_SIZE - 1) / (BLOCK_SIZE)) N_work += i;
    cudaMalloc(&extra_d, N_work * n * sizeof(double)); // extra memory buffer for reduction across thread-blocks
    cudaDeviceSynchronize();


    tick = omp_get_wtime();

    dim3 block_dim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid_dim(n / BLOCK_SIZE, n / BLOCK_SIZE);
    // map
    gpu_map_vec_mat_mul <<< grid_dim, block_dim >>> (mat_d, vec_d, temp_mat_d, n);
    cudaDeviceSynchronize();
    Check_CUDA_Error("map failed");

    // reduce
    double *sum_d = extra_d; // for reduction intermediate number
    long Nb = (n + BLOCK_SIZE - 1) / (BLOCK_SIZE);
    gpu_reduce_vec_mat_mul <<< grid_dim, block_dim >>> (sum_d, temp_mat_d, n);
    Check_CUDA_Error("first reduce failed");
    printf("stop 2");
    while (Nb > 1) {
        long next_buffer_offset = Nb * n;
        Nb = (Nb + BLOCK_SIZE - 1) / (BLOCK_SIZE);
        dim3 cur_grid(n / BLOCK_SIZE, Nb / BLOCK_SIZE);
        Check_CUDA_Error("some reduce failed");
        gpu_reduce_vec_mat_mul << < cur_grid, block_dim >> > (sum_d + next_buffer_offset, sum_d, Nb);
        sum_d += next_buffer_offset; // currently sum_d point to reduction result
    }

    // fetch mul result
    // currently sum_d is out anwser
    cudaMemcpyAsync(&vec_mul, sum_d, n * sizeof(double), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    Check_CUDA_Error("copy result back failed");

    time = omp_get_wtime() - tick;
    printf("GPU benchmark\n");
    printf("Time = %f\n", time);
    printf("GPU Bandwidth = %f GB/s\n", 2 * n * sizeof(double) / time / 1e9);
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

