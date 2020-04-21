#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <omp.h>
#include <algorithm>

int N = 200;
int SIZE = N+2; // always N+2
int MAT_SIZE = SIZE*SIZE;
long maxIter = INT32_MAX;
double h, hSqr, hSqrInverse;

inline double sqr(double x) {
    return x*x;
}

double calcResidual(const double* u) {
    // since f is always 1, just hardcoded 1 into formula
    // N is actually N+2 (0-N+1)
    double res = 0;
    for (int i = 1; i <= N; ++i) {
        for (int j = 1; j <= N; ++j) {
            res += sqr((-u[(i-1)*SIZE+j]-u[i*SIZE+j-1]+4*u[i*SIZE+j]-u[(i+1)*SIZE+j]-u[i*SIZE+j+1]) * hSqrInverse - 1);
        }
    }
    return std::sqrt(res);
}

/**
 * return the iteration of jacobi
 * @return
 */
long jacobi_cpu(double* u, double* v) {
    double initResidual = calcResidual(u);
    double curResidual = 0;
    long k;
    for (k = 1; k <= maxIter; ++k) {
#   pragma omp parallel for num_threads(6)
        for (int i = 1; i <= N; ++i) {
            for (int j = 1; j <= N; ++j) {
                // update u
                v[i*SIZE+j] = (hSqr+u[(i-1)*SIZE+j]+u[i*SIZE+j-1]+u[(i+1)*SIZE+j]+u[i*SIZE+j+1])/4;
            }
        }
        std::swap(u, v);
        curResidual = calcResidual(u);

        if (initResidual/curResidual > 1e+6) {
            break;
        }
    }
    return k;
}

#define TILE_LEN 16 // block size be 8*8=64

__device__ double gpu_residual;

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

__global__ void gpu_residual_calc(const double* u, int n, double _hsqrinverse) {
    __shared__ double smem[TILE_LEN][TILE_LEN];
    int i = (threadIdx.x + 1) + blockIdx.x*blockDim.x;
    int j = (threadIdx.y + 1) + blockIdx.y*blockDim.y;

    int size = n-2;
    if(i <= n && j <= n){
        double diff = (-u[(i-1)*size+j]-u[i*size+j-1]+4*u[i*size+j]-u[(i+1)*size+j]-u[i*size+j+1]) * _hsqrinverse - 1;
        diff = std::sqrt(diff);
        smem[threadIdx.x][threadIdx.y] = diff;
        __syncthreads();
    }

    if (threadIdx.y == 0) {
        double acc = 0;
        for (int k = 0; k < TILE_LEN; k++) {
            acc += smem[threadIdx.x][threadIdx.y];
        }
        smem[threadIdx.x][0] = acc;
        __syncthreads();
    }

    if (threadIdx.x == 0 && threadIdx.y == 0) {
        double acc = 0;
        for (int k = 0; k < TILE_LEN; k++) {
            acc += smem[threadIdx.x][0];
        }
        atomicAdd2(&gpu_residual, acc);
    }
}

__global__ void gpu_jacobi(double* u, double* v, double hsqr, int size) {
    int i = (threadIdx.x + 1) + blockIdx.x*blockDim.x;
    int j = (threadIdx.y + 1) + blockIdx.y*blockDim.y;
    v[i*size+j] = (hsqr+u[(i-1)*size+j]+u[i*size+j-1]+u[(i+1)*size+j]+u[i*size+j+1])/4;
}


int main(int argc, char** argv) {
    printf("Jacobi 2D\n");
    printf("=====================\n");

    h = 1./(double) (N+1);
    hSqr = h*h;
    hSqrInverse = 1/hSqr;

    auto* u = new double[MAT_SIZE];
    auto* v = new double[MAT_SIZE];
    // initialization
    for (int i = 0; i < SIZE*SIZE; ++i) {
        u[i] = 0;
        v[i] = 0;
    }

    double tick = omp_get_wtime();
    double tok;
    // jacobi start
    long cpu_iter = jacobi_cpu(u, v);
    tok = omp_get_wtime();
    printf("Openmp cpu\n");
    printf("Used time: %lf \n Iteration: %ld\n", (tok-tick), cpu_iter);

    printf("=====================\n");

    // gpu

//    // allocate
    std::fill(u, u+MAT_SIZE, 0);
    std::fill(v, v+MAT_SIZE, 0);
    double* u_d;
    double* v_d;
    cudaMalloc(&u_d, MAT_SIZE * sizeof(double));
    cudaMalloc(&v_d, MAT_SIZE * sizeof(double));
    cudaMemcpyAsync(u_d, u, MAT_SIZE * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(v_d, v, MAT_SIZE * sizeof(double), cudaMemcpyHostToDevice);

    dim3 grid(N/TILE_LEN, N/TILE_LEN);
    dim3 block(TILE_LEN, TILE_LEN);

    tick = omp_get_wtime();
    long gpu_iter = 0;
    double init_res;
    gpu_residual_calc<<<grid, block>>>(u_d, N, hSqrInverse);
    cudaMemcpyToSymbol(gpu_residual, &init_res, sizeof(double));
    double cur_res = 0;
    while (gpu_iter < maxIter) {
        gpu_jacobi<<<grid, block>>>(u_d, v_d, N, hSqr);
        std::swap(u_d, v_d);
        gpu_residual_calc<<<grid, block>>>(u_d, N, hSqrInverse);
        cudaMemcpyToSymbol(gpu_residual, &cur_res, sizeof(double));
        if (init_res/cur_res > 1e+6) {
            break;
        }
    }

    tok = omp_get_wtime();
    printf("GPU\n");
    printf("Used time: %lf \n Iteration: %ld\n", (tok-tick), gpu_iter);

    free(u);
    free(v);
    cudaFree(u_d);
    cudaFree(v_d);
}