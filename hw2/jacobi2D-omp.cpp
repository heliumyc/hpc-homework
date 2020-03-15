//
// Created by CONG YU on 2020/3/14.
//
//#define DEBUG
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include "utils.h"

long N, maxIter;
int threadNum;
double h, hSqr, hSqrInverse;
long SIZE; // always N+2

using namespace std;

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
    return sqrt(res);
}


int main(int argc, char** argv) {
    // check args
    if (argc == 4) {
        N = (long) strtol(argv[1], nullptr, 10);
        SIZE = N+2;
        maxIter = (long) strtol(argv[2], nullptr, 10);
        if (maxIter == -1) {
            maxIter = INT32_MAX;
        }
        threadNum = (int) strtol(argv[3], nullptr, 10);
    } else {
        printf("usage: ./program N iteration(-1 nonstop) thread_number\n");
        exit(0);
    }

    h = 1./(double) (N+1);
    hSqr = h*h;
    hSqrInverse = 1/hSqr;

    auto* u = new double[SIZE*SIZE];
    auto* v = new double[SIZE*SIZE];
    // initialization
    for (int i = 0; i < SIZE*SIZE; ++i) {
        u[i] = 0;
        v[i] = 0;
    }

    double initResidual = calcResidual(u);
    double curResidual = 0;
    Timer timer;
    timer.tic();
    // jacobi start
    int k;
    for (k = 1; k <= maxIter; ++k) {
        for (int i = 1; i <= N; ++i) {
            for (int j = 1; j <= N; ++j) {
                // update u
                v[i*SIZE+j] = (hSqr+u[(i-1)*SIZE+j]+u[i*SIZE+j-1]+u[(i+1)*SIZE+j]+u[i*SIZE+j+1])/4;
            }
        }
        swap(u, v);
        curResidual = calcResidual(u);

#ifdef DEBUG
        printf("round %d with residual: %lf\n", k, curResidual);
#endif
        if (initResidual/curResidual > 1e+6) {
            break;
        }
    }
    double endTime = timer.toc();
    printf("Used time: %lf \n Iteration: %d\n", endTime, k);
}
