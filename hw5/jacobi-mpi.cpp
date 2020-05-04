//
// Created by CONG YU on 5/3/20.
//

/* MPI-parallel Jacobi smoothing to solve -u''=f
 * Global vector has N unknowns, each processor works with its
 * part, which has lN = N/p unknowns.
 * Author: Georg Stadler
 */
#include <stdio.h>
#include <math.h>
#include <mpi.h>
#include <string.h>
#include <cstdlib>
#include <stdlib.h>
#include <algorithm>
#include <iostream>

inline double sqr(double x) {return x*x;}

/* compuate global residual, assuming ghost values are updated */
double compute_residual(double *lu, int lN, double invhsq) {
    double tmp, gres = 0.0, lres = 0.0;
    int SIZE = lN+2;

    for (int i = 1; i <= lN; i++){
        for (int j = 1; j <= lN; j++) {
            lres += sqr((-lu[(i-1)*SIZE+j]-lu[i*SIZE+j-1]+4*lu[i*SIZE+j]-lu[(i+1)*SIZE+j]-lu[i*SIZE+j+1]) * invhsq - 1);
        }
    }
    /* use allreduce for convenience; a reduce would also be sufficient */
    MPI_Allreduce(&lres, &gres, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    return sqrt(gres);
}

int main(int argc, char * argv[]) {
    int mpirank, p, N, lN, iter, max_iters, size;
    MPI_Status status;
    MPI_Request request_out[4];
    MPI_Request request_in[4];

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpirank);
    MPI_Comm_size(MPI_COMM_WORLD, &p);

    /* get name of host running MPI process */
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;
//    MPI_Get_processor_name(processor_name, &name_len);
//    printf("Rank %d/%d running on %s.\n", mpirank, p, processor_name);

    if (argc != 3) {
        printf("-lN -maxiter\n");
        MPI_Abort(MPI_COMM_WORLD, 0);
    }
    sscanf(argv[1], "%d", &lN);
    sscanf(argv[2], "%d", &max_iters);

    MPI_Comm_size(MPI_COMM_WORLD, &p);

    int pInRowNum = (int) sqrt(p);
    if (pInRowNum*pInRowNum != p && mpirank == 0) {
        printf("p must be square number");
        MPI_Abort(MPI_COMM_WORLD, 0);
    }

    N = lN*pInRowNum;

    /* compute number of unknowns handled by each process */
//    lN = N / pInRowNum;
    size = lN+2;
    if ((N % pInRowNum != 0) && mpirank == 0 ) {
        printf("N: %d, local N: %d\n", N, lN);
        printf("Exiting. N must be a multiple of sqrt(p)\n");
        MPI_Abort(MPI_COMM_WORLD, 0);
    }

    /* timing */
    MPI_Barrier(MPI_COMM_WORLD);
    double tt = MPI_Wtime();

    /* Allocation of vectors, including left and right ghost points */
    double * lu    = (double *) malloc(size*size*sizeof(double));
    double * lunew = (double *) malloc(size*size*sizeof(double));

    std::fill(lunew, lunew+size*size, 0);

    // temp buffer for ghost nodes nearby
    // 0 for left, 1 for right
    double buffer_out[2][lN+2];
    double buffer_in[2][lN+2];
    for (int k = 0; k < 2; k++) {
        std::fill(buffer_out[k], buffer_out[k] + (lN + 2), 0);
        std::fill(buffer_in[k], buffer_in[k] + (lN + 2), 0);
    }

    double h = 1.0 / (N + 1);
    double hsq = h * h;
    double invhsq = 1./hsq;
    double gres, gres0, tol = 1e-5;

    /* initial residual */
    gres0 = compute_residual(lu, lN, invhsq);
    gres = gres0;

    int row, col;
    for (iter = 0; iter < max_iters && gres/gres0 > tol; iter++) {
        /* interleaf computation and communication: compute the first
         * and last value, which are communicated with non-blocking
         * send/recv. During that communication, do all the local work */

        // row and col of blocks
        row = mpirank/pInRowNum;
        col = mpirank%pInRowNum;
//        std::cout << row << " " << col << std::endl;

        /* Jacobi step for boundary points */
        for (int i = 1; i <= lN; i++) {
            // left
            lunew[1+i*size] = (hsq+lu[0+i*size]+lu[2+i*size]+lu[1+(i-1)*size]+lu[1+(i+1)*size])/4;
            // right
            lunew[lN+i*size] = (hsq+lu[lN-1+i*size]+lu[lN+1+i*size]+lu[lN+(i-1)*size]+lu[lN+(i+1)*size])/4;
            // top
            lunew[i+lN*size] = (hsq+lu[i-1+lN*size]+lu[i+1+lN*size]+lu[i+(lN-1)*size]+lu[i+(lN+1)*size])/4;
            // bottom
            lunew[i+1*size] = (hsq+lu[i-1+size]+lu[i+1+size]+lu[i]+lu[i+2*size])/4;
        }

        if (row > 0) {
            // read buffer from bottom
//            std::cout << "Rank in " << col+(row-1)*pInRowNum << std::endl;
//            std::cout << "Rank out " << col+(row-1)*pInRowNum << std::endl;
            MPI_Irecv(lunew+1, lN, MPI_DOUBLE, col+(row-1)*pInRowNum, 904, MPI_COMM_WORLD, &request_in[4]);
            MPI_Isend(lunew+size+1, lN, MPI_DOUBLE, col+(row-1)*pInRowNum, 903, MPI_COMM_WORLD, &request_out[3]);
        }

        if (row < pInRowNum-1) {
            // read buffer from top
//            std::cout << "Rank in " << col+(row+1)*pInRowNum << std::endl;
//            std::cout << "Rank out " << col+(row+1)*pInRowNum << std::endl;
            MPI_Irecv(lunew+(size-1)*size, lN, MPI_DOUBLE, col+(row+1)*pInRowNum, 903, MPI_COMM_WORLD, &request_in[3]);
            MPI_Isend(lunew+size*size-1, lN, MPI_DOUBLE, col+(row+1)*pInRowNum, 904, MPI_COMM_WORLD, &request_out[4]);
        }

        if (col > 0) {
            // load current data to left buffer and send to neighbor
            for (int j = 1; j <= lN; j++) {
                buffer_out[0][j] = lunew[1+j*size];
            }
//            std::cout << "Rank in " << col-1+row*pInRowNum << std::endl;
//            std::cout << "Rank out " << col-1+row*pInRowNum << std::endl;
            MPI_Irecv(buffer_in[0]+1, lN, MPI_DOUBLE, col-1+row*pInRowNum, 901, MPI_COMM_WORLD, &request_in[1]);
            MPI_Isend(buffer_out[0]+1, lN, MPI_DOUBLE, col-1+row*pInRowNum, 900, MPI_COMM_WORLD, &request_out[0]);
        }

        if (col < pInRowNum-1) {
            // load current data to right buffer and send to neighbor
            for (int j = 1; j <= lN; j++) {
                buffer_out[1][j] = lunew[lN+j*size];
            }
//            std::cout << "Rank in " << col+1+row*pInRowNum << std::endl;
//            std::cout << "Rank out " << col+1+row*pInRowNum << std::endl;
            MPI_Irecv(buffer_in[1]+1, lN, MPI_DOUBLE, col+1+row*pInRowNum, 900, MPI_COMM_WORLD, &request_in[0]);
            MPI_Isend(buffer_out[1]+1, lN, MPI_DOUBLE, col+1+row*pInRowNum, 901, MPI_COMM_WORLD, &request_out[1]);
        }

        /* Jacobi step for all the inner points */
        for (int i = 2; i < lN; i++){
            for (int j = 2; j < lN; j++) {
                lunew[i+j*size] = (hsq+lu[i-1+j*size]+lu[i+1+j*size]+lu[i+(j-1)*size]+lu[i+(j+1)*size])/4;
            }
        }

        /* check if Isend/Irecv are done */
        if (row > 0) {
            MPI_Wait(&request_in[4], &status);
            MPI_Wait(&request_out[3], &status);
        }

        if (row < pInRowNum-1) {
            MPI_Wait(&request_in[3], &status);
            MPI_Wait(&request_out[4], &status);
        }

        if (col > 0) {
            MPI_Wait(&request_in[1], &status);
            MPI_Wait(&request_out[0], &status);
        }

        if (col < pInRowNum-1) {
            MPI_Wait(&request_in[0], &status);
            MPI_Wait(&request_out[1], &status);
        }

        // load buffer in to the ghost points
        for (int i = 1; i <= lN; i++) {
            // left
            lunew[i*size] = buffer_in[0][i];
            // right
            lunew[size-1+i*size] = buffer_in[1][i];
        }

        /* copy newu to u using pointer flipping */
        std::swap(lunew, lu);
        if (0 == (iter % 100)) {
            gres = compute_residual(lu, lN, invhsq);
            if (0 == mpirank) {
                printf("Iter %d: Residual: %g\n", iter, gres);
            }
        }
    }

    /* Clean up */
    free(lu);
    free(lunew);

    /* timing */
    MPI_Barrier(MPI_COMM_WORLD);
    double elapsed = MPI_Wtime() - tt;
    if (0 == mpirank) {
        printf("Time elapsed is %f seconds.\n", elapsed);
    }
    MPI_Finalize();
//    return 0;
}


