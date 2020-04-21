#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <omp.h>

long N = 50;
long SIZE = N+2; // always N+2
long maxIter = 30000000;
double h, hSqr, hSqrInverse;

inline double sqr(double x) {
    return x*x;
}

void Check_CUDA_Error(const char *message) {
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        fprintf(stderr, "ERROR: %s: %s\n", message, cudaGetErrorString(error));
        exit(-1);
    }
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
        double* temp = u;
        u = v;
        v = temp;
        curResidual = calcResidual(u);

        if (initResidual/curResidual > 1e+6) {
            break;
        }
    }
    return k;
}

#define TILE_LEN 8 // block size be 8*8=64

//__device__ double gpu_residual;

//__device__ double atomicAdd2(double* address, double val)
//{
//    auto* address_as_ull =
//            (unsigned long long int*)address;
//    unsigned long long int old = *address_as_ull, assumed;
//
//    do {
//        assumed = old;
//        old = atomicCAS(address_as_ull, assumed,
//                        __double_as_longlong(val +
//                                             __longlong_as_double(assumed)));
//
//        // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
//    } while (assumed != old);
//
//    return __longlong_as_double(old);
//}
//
//__global__ void gpu_residual_calc(const double* u, const double* f, long n) {
//    double res = 0;
//
//    __shared__ double smem[TILE_LEN][TILE_LEN];
//
//    long idx = (threadIdx.x + 1) + blockIdx.x*blockDim.x;
//    long idy = (threadIdx.y + 1) + blockIdx.y*blockDim.y;
//
//    if(idx < n && idy < n){
//        double diff = f[idx*n + idy] - u[idx*n + idy];
//        smem[threadIdx.x][threadIdx.y] = diff > 0? diff: -diff; // abs diff
//        __syncthreads();
//    }
//
//    for (int i = 1; i <= N; ++i) {
//        for (int j = 1; j <= N; ++j) {
//            res += sqr((-u[(i-1)*SIZE+j]-u[i*SIZE+j-1]+4*u[i*SIZE+j]-u[(i+1)*SIZE+j]-u[i*SIZE+j+1]) * hSqrInverse - 1);
//        }
//    }
//    gpu_residual = std::sqrt(res);
//}



int main(int argc, char** argv) {
    printf("Jacobi 2D\n");
    printf("=====================\n");

    h = 1./(double) (N+1);
    hSqr = h*h;
    hSqrInverse = 1/hSqr;

    double* u = (double*) malloc(SIZE*SIZE*sizeof(double));
    double* v = (double*) malloc(SIZE*SIZE*sizeof(double));
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
    for (int i = 0; i < SIZE*SIZE; ++i) {
        u[i] = 0;
        v[i] = 0;
    }
    free(u);
    free(v);
    double* uu = (double*) malloc(SIZE*SIZE*sizeof(double));;
    printf("alloc1 host done");
    double* vv = (double*) malloc(SIZE*SIZE*sizeof(double));;
    printf("alloc2 host done");
    for (int k = 0; k < SIZE*SIZE; ++k) {
        if (k > 10000)
            printf("fuc");
        uu[k] = 0;
        vv[k] = 0;
    }

    double* u_d;
    double* v_d;
    cudaMalloc(&u_d, SIZE*SIZE * sizeof(double));
    Check_CUDA_Error("alloc cuda failed");
    cudaMalloc(&v_d, SIZE*SIZE * sizeof(double));
    Check_CUDA_Error("alloc cuda failed");
    cudaMemcpyAsync(u_d, uu, SIZE*SIZE * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(v_d, vv, SIZE*SIZE * sizeof(double), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    Check_CUDA_Error("alloc cuda failed");
    printf("alloc cuda done");

}
