#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <iostream>
#include <cmath>
#include "utils.h"
#define block_size 1024
__device__ double production;

__device__ double atomicAdd2(double* address, double val)
{
    unsigned long long int* address_as_ull =
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

double vec_inner_product_cpu(const double* va, const double* vb, long size){
    double result = 0.0;
#pragma omp parallel for shared(va, vb) reduction(+: result)
    for(long i = 0; i < size; ++i){
        result += va[i] * vb[i];
    }

    return result;
}

void vec_mat_mul_cpu(const double* mat, const double* vec, double* ret, long rows, long columns){

#pragma omp parallel for shared(mat, vec, ret)
    for(long i = 0; i < rows; ++i){
        double tmp = 0.0;
        for(long j = 0; j < columns; j++){
            tmp += mat[i*columns + j] * vec[j];
        }
        ret[i] = tmp;
    }
}

__global__
void vec_inner_product_kernel(const double* va, const double* vb, long size){
    extern __shared__ double shared_cache[];
    long id = threadIdx.x + blockIdx.x * blockDim.x;

    if(id < size){
        shared_cache[threadIdx.x] = va[id] * vb[id];
        __syncthreads();

        // perform reduction in block
        for (unsigned int s = blockDim.x /2; s>0; s >>=1) {
            if (threadIdx.x < s) {
                shared_cache[threadIdx.x] += shared_cache[threadIdx.x + s];
            }
            __syncthreads();
        }

        // add the first element of shared_cache of each block to production
        if(threadIdx.x == 0){
            atomicAdd2(&production, shared_cache[0]);
        }
    }
}

__global__
void vec_mat_mul_kernel(const double* va, const double* vb, double* ret,  long rows, long columns){
    extern __shared__ double shared_cache[];

    long idx = threadIdx.x + blockIdx.x*blockDim.x;
    long idy = threadIdx.y + blockIdx.y*blockDim.y;

    if(idx < rows && idy < columns){
        long tindex = idx*columns + idy;
        shared_cache[threadIdx.x * blockDim.y + threadIdx.y] = va[tindex] * vb[idy];
        __syncthreads();
    }

    for (unsigned int s = blockDim.y /2; s>0; s >>=1) {
        if (threadIdx.y < s) {
            shared_cache[threadIdx.x * blockDim.y + threadIdx.y] += shared_cache[threadIdx.x * blockDim.y + threadIdx.y + s];
        }
        __syncthreads();
    }

    if(threadIdx.y == 0){
        atomicAdd2(ret + idx, shared_cache[threadIdx.x * blockDim.y + threadIdx.y]);
    }
}

// __global__
// void vec_mat_mul_kernel(const double* va, const double* vb, double* ret,  long rows, long columns){
//     long idx = threadIdx.x + blockIdx.x*blockDim.x;
//     if(idx < rows){
//         double tmp = 0.0;

//         for(int i = 0; i < columns; ++i)
//             tmp += va[idx * columns + i] * vb[i];

//         ret[idx] = tmp;
//     }
//     __syncthreads();

// }

int main(){

    /*
        Implementation of calculating vector inner product
    */
    printf("VECTOR INNER PRODUCT:\n");
    long N = 1UL<<21;
    // host pointers and placeholder for result
    double* host_a = (double*) malloc(N * sizeof(double));
    double* host_b = (double*) malloc(N * sizeof(double));
    double production_cpu;

    // the following part is adapted from gpu03.cu, random initialize 2 different vector
#pragma omp parallel for schedule(static)
    for (long i = 0; i < N; i++) {
        host_a[i] = 1.0 / (rand() % 2 + 1);
        host_b[i] = 1.5 / (rand() % 3 + 2);
    }

    double tt = omp_get_wtime();
    Timer T;
    T.tic();
    production_cpu = vec_inner_product_cpu(host_a, host_b, N);
    printf("CPU Bandwidth = %f GB/s\n", 2*N*sizeof(double) / (omp_get_wtime() - tt) / 1e9);

    // device pointers and placeholder for result
    double* d_va;
    double* d_vb;
    double production_gpu = 0.0;

    cudaMalloc(&d_va, N*sizeof(double));
    cudaMalloc(&d_vb, N*sizeof(double));

    // copy 2 vectors into gpu memory
    cudaMemcpy(d_va, host_a,  N*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vb, host_b,  N*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(production, &production_gpu, sizeof(double));

    int grid_size = (int) std::ceil((double) N / block_size);

    tt = omp_get_wtime();
    vec_inner_product_kernel<<<grid_size, block_size, block_size*sizeof(double)>>>(d_va, d_vb, N);
    cudaDeviceSynchronize();
    printf("GPU Bandwidth = %f GB/s\n", 2*N*sizeof(double) / (omp_get_wtime()-tt)/1e9);
    cudaMemcpyFromSymbol(&production_gpu, production, sizeof(double));

    // check the error
    printf("CPU Production: %f \n", production_cpu);
    printf("GPU Production: %f \n", production_gpu);
    printf("Error: %f\n", production_cpu - production_gpu);

    // free all allocated memory
    free(host_a);
    free(host_b);

    cudaFree(d_va);
    cudaFree(d_vb);
    // =============== end of vector inner product =====================


    /*
        Implementation of vector-matrix multplication
    */
    printf("\nVECTOR MATRIX MULTIPLICATION:\n");
    long rows = 1UL<<19, columns = 1UL<<8;

    // host pointers and placeholder for result
    double* host_mat = (double*) malloc(rows * columns * sizeof(double));
    double* host_vec = (double*) malloc(columns * sizeof(double));
    double* cpu_vec_ret = (double*) malloc(rows * sizeof(double));

    // the following part is adapted from gpu03.cu, random initialize 2 different vector
#pragma omp parallel for schedule(static)
    for (long i = 0; i < columns*rows; i++) {
        host_mat[i] = 1.0 / (rand() % 2 + 1);
    }

#pragma omp parallel for schedule(static)
    for (long i = 0; i < columns; i++) {
        host_vec[i] = 1.0 / (rand() % 3 + 1);
        cpu_vec_ret[i] = 0.0;
    }

    tt = omp_get_wtime();
    vec_mat_mul_cpu(host_mat, host_vec, cpu_vec_ret, rows, columns);
    printf("CPU Bandwidth = %f GB/s\n", (2*rows*columns + rows)*sizeof(double) / (omp_get_wtime() - tt) / 1e9);

    // host pointers and placeholder for result
    double* d_mat;
    double* d_vec;
    double* gpu_vec_ret;
    cudaMalloc(&d_mat, rows * columns * sizeof(double));
    cudaMalloc(&d_vec, columns * sizeof(double));
    cudaMalloc(&gpu_vec_ret, rows * sizeof(double));

    double* host_gpu_ret = (double*) malloc(rows * sizeof(double));
    for(int i = 0; i < rows; ++i)
        host_gpu_ret[i] = 0.0;

    cudaMemcpy(d_mat, host_mat,  rows*columns*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vec, host_vec,  columns*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_vec_ret, host_gpu_ret, rows*sizeof(double), cudaMemcpyHostToDevice);

    tt = omp_get_wtime();
    // grid_size = (int) std::ceil((double) rows / columns);
    // vec_mat_mul_kernel<<<grid_size, columns>>>(d_mat, d_vec, gpu_vec_ret, rows, columns);

    dim3 dimBlock(8, 8);
    dim3 dimGrid(ceil(rows / dimBlock.x), ceil(columns / dimBlock.y));
    vec_mat_mul_kernel<<<dimGrid, dimBlock, dimBlock.x*dimBlock.y*sizeof(double)>>>(d_mat, d_vec, gpu_vec_ret, rows, columns);
    cudaDeviceSynchronize();
    cudaMemcpy(host_gpu_ret, gpu_vec_ret, rows*sizeof(double), cudaMemcpyDeviceToHost);
    printf("GPU Bandwidth = %f GB/s\n", (2*rows*columns + rows)*sizeof(double) / (omp_get_wtime()-tt)/1e9);
    printf("The First Element of CPU ver: %f\n", cpu_vec_ret[0]);
    printf("The First Element of GPU ver: %f\n", host_gpu_ret[0]);

    double max_error = 0.0;
    for(int i = 0; i < rows; ++i){
        max_error = max(max_error, std::abs(host_gpu_ret[i] - cpu_vec_ret[i]));
    }

    printf("Max Error: %f\n", max_error);

    free(host_mat);
    free(host_vec);
    free(cpu_vec_ret);

    cudaFree(d_mat);
    cudaFree(d_vec);
    cudaFree(gpu_vec_ret);
}
