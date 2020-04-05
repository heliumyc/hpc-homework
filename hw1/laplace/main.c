#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <stdint.h>

int LOG_RESIDUAL = 0;
double INIT_RESIDUAL = 0;

int N = 100;
int maxIter = 100;
double h = 1;
int method = 1;
double H; // 1/h^2
double H_inverse;

double calcRes(const double*u, const double *f) {
    double res = 0, delta = 0;
    delta = f[0] - (2*H*u[0]-1*H*u[1]);
    res += delta*delta;
    delta = f[N-1] - (-1*H*u[N-2]+2*H*u[N-1]);
    res += delta*delta;
    for (int i = 1; i < N-1; i++) {
        delta = f[i] - (-1*H*u[i-1]+2*H*u[i]-1*H*u[i+1]);
        res += delta*delta;
    }
    res = sqrt(res);
    // print
    return res;
}

int jacobi(double *u, const double *f) {
    double res;
    double lastNewVal, curNewVal;
    for (int k = 0; k < maxIter; k++) {
        lastNewVal = (H_inverse + u[1])/2;
        for (int i = 1; i < N-1; i++) {
            // (1+H*u[i-1]+H*u[i+1])/(2*H)
            curNewVal = (H_inverse + u[i-1]+u[i+1])/2;
            u[i-1] = lastNewVal;
            lastNewVal = curNewVal;
        }
        u[N-1] = (H_inverse+u[N-2])/2;
        u[N-2] = lastNewVal;
        res = calcRes(u, f);
        if (LOG_RESIDUAL) printf("iteration %d\tresidual %f\t"
                                 "decreased %f\tdecreased factor %f\n",
                                 k+1, res, (INIT_RESIDUAL - res), INIT_RESIDUAL/res);
        if (INIT_RESIDUAL/res >= 1e6) {
            return k+1;
        }
    }
    return maxIter;
}

int gauss(double *u, const double *f) {
    double res;
    for (int k = 0; k < maxIter; k++) {
        u[0] = (H_inverse + u[1])/2;
        for (int i = 1; i < N-1; i++) {
            u[i] = (H_inverse + u[i-1] + u[i+1])/2;
        }
        u[N-1] = (H_inverse + u[N-2])/2;
        res = calcRes(u, f);
        if (LOG_RESIDUAL) printf("iteration %d\tresidual %f\t"
                                 "decreased %f\tdecreased factor %f\n",
                                 k+1, res, (INIT_RESIDUAL - res), INIT_RESIDUAL/res);
        if (INIT_RESIDUAL/res >= 1e6) {
            return k+1;
        }
    }
    return maxIter;
}

int main(int argc, char **argv) {

    // read args from cmd
    /*
     * input is as follow
     * ./program N MaxIteration Method (1 -> Jacobi, 2 -> Gauss) [LOG-RESIDUAL]
     */
    if (argc == 4 || argc == 5) {
        N = (int) strtol(argv[1], NULL, 10);
        maxIter = (int) strtol(argv[2], NULL, 10);
        if (maxIter == -1) {
            maxIter = INT32_MAX;
        }
        method = (int) strtol(argv[3], NULL, 10);
        if (argc == 5) {
            LOG_RESIDUAL = (int) strtol(argv[4], NULL, 10) == 0? 0 : 1;
        }
    } else {
        printf("usage: ./program N {MaxIteration | -1 (for non stop)}"
               " {1 (for Jacobi) | 2 (for Gauss)} [Log-Residual = 0]\n");
        exit(0);
    }

    h = 1.0/(N+1);
    H = (N+1)*(N+1);
    H_inverse = 1.0 / H;

    // malloc
    double *u = malloc(N* sizeof(double));
    double *f = malloc(N* sizeof(double));

    memset(u, 0.0, N*sizeof(double));
    // init f
    for (int i = 0; i < N; ++i) {
        f[i] = 1;
    }

    INIT_RESIDUAL = calcRes(u, f);
    int iter = 0;
    clock_t s = clock();
    if (method == 1) {
        iter = jacobi(u, f);
    } else {
        iter = gauss(u, f);
    }

    clock_t t = clock();

    printf("iteration runs: %d\n", iter);
    printf("time used: %.4lf s\n", (double) ( t-s )/CLOCKS_PER_SEC);

    // free
    free(u);
    free(f);

    return 0;
}
