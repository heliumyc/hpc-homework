//
// Created by CONG YU on 4/20/20.
//

#include <algorithm>
#include <stdio.h>
#include <omp.h>
#include <random>
#include <string>

double compare_vec(double* v1, double* v2, long n) {
    int diff = 0;
#pragma omp parallel for reduction (+: diff)
    for (int i = 0; i < n; i++) {
        diff += std::abs(v1[i]-v2[i]);
    }
    return diff;
}

void sequential_vec_mat_mul(double* res, const double* mat, const double* vec, long n) {
    for (long i = 0; i < n; i++) {
        double sum = 0;
        for (int j = 0; j < n; j++) {
            sum += vec[j]*mat[j+i*n];
        }
        res[i] = sum;
    }
}

void openmp_vec_mat_mul(double* res, const double* mat, const double* vec, long n) {
#pragma omp parallel for schedule(static)
    for (long i = 0; i < n; i++) {
        double sum = 0;
#pragma omp parallel for reduction (+: sum)
        for (int j = 0; j < n; j++) {
            sum += vec[j]*mat[j+i*n];
        }
        res[i] = sum;
    }
}

int main() {

    long n = 1<<12;
    double* vec = (double *) malloc(n * sizeof(double));
    double* mat = (double *) malloc(n*n * sizeof(double));
    double* vec_ref = (double *) malloc(n * sizeof(double));
    double* vec_omp = (double *) malloc(n * sizeof(double));

    // random
    std::random_device rd;
    std::default_random_engine gen(rd());
    std::uniform_real_distribution<double> uniformRealDistribution(-1, 1);

    for (long i; i < n; i++) {
        vec[i] = uniformRealDistribution(gen);
        for (long j; j < n; j++) {
            mat[j+i*n] = uniformRealDistribution(gen);
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
    printf("CPU Bandwidth = %f GB/s\n", (n*n+n)*sizeof(double) / time/1e9);
    printf("Error = %f\n", compare_vec(vec_ref, vec_ref, n));

    // openmp calculation
    tick = omp_get_wtime();
    openmp_vec_mat_mul(vec_omp, mat, vec, n);
    time = omp_get_wtime() - tick;
    printf("Openmp benchmark\n");
    printf("Time = %f\n", time);
    printf("CPU Bandwidth = %f GB/s\n", (n*n+n)*sizeof(double) / time/1e9);
    printf("Error = %f\n", compare_vec(vec_ref, vec, n));

    printf("------------\n");
}

