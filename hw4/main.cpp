#include <iostream>
#include <algorithm>
#include <stdio.h>
#include <omp.h>
#include <random>
#include <string>

// cpu sequential computation
void sequential_vec_inner_product(double *res, const double *a, const double *b, long n) {
    double acc = 0;
    for (long i = 0; i < n; i++) {
        acc += a[i] * b[i];
    }
    *res = acc;
}

void inner_product(double *res, const double *a, const double *b, long n) {
    double sum = 0;
#pragma omp parallel for reduction (+: sum)
    for (long i = 0; i < n; i++) {
        sum += a[i] * b[i];
    }
    *res = sum;
}

int main() {
    long n = (1UL << 25); // 2^25

    // malloc
    auto *a = (double *) malloc(n * sizeof(double));
    auto *b = (double *) malloc(n * sizeof(double));
    auto *temp = (double *) malloc(n * sizeof(double));

    // random
    std::random_device rd;
    std::default_random_engine gen(rd());
    std::uniform_real_distribution<double> uniformRealDistribution(-1, 1);

    // init
    omp_set_num_threads(6);
//#pragma omp parallel for schedule(static)
    for (long i = 0; i < n; i++) {
        a[i] = uniformRealDistribution(gen);
        b[i] = uniformRealDistribution(gen);
    }

    double time;
    double tick;

    tick = omp_get_wtime();
    double ref;
    sequential_vec_inner_product(&ref, a, b, n);
    time = omp_get_wtime() - tick;
    printf("Sequential benchmark\n");
    printf("Time = %f\n", time);
    printf("CPU Bandwidth = %f GB/s\n", 2 * n * sizeof(double) / time / 1e9);
    printf("Error = %f\n", std::abs(ref - ref));

    printf("------------\n");

    // openmp calculation
    tick = omp_get_wtime();
    double openmp_res;
    inner_product(&openmp_res, a, b, n);
    time = omp_get_wtime() - tick;
    printf("Openmp benchmark\n");
    printf("Time = %f\n", time);
    printf("CPU Bandwidth = %f GB/s\n", 2 * n * sizeof(double) / time / 1e9);
    printf("Error = %f\n", std::abs(openmp_res - openmp_res));

    printf("------------\n");

    // free
    free(a);
    free(b);
    free(temp);
}
