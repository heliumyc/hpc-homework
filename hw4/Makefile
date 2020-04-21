CFLAGS:=-O3  # -g = debug, -O2 for optimized code CPPFLAGS for g++
CC:=nvcc
OMP:=-fopenmp
STD:=-std=c++11

.PHONY: all

all:  mmult ompbugs jacobi gs;


mulvec:
	$(CC) $(CFLAGS) $(STD) $(OMP) -o vec_mat_mul vec_mat_mul.cu

innerproduct:
	$(CC) $(CFLAGS) $(STD) $(OMP) -o inner_product inner_product.cu

jacobi:
	$(CC) $(CFLAGS) $(STD) $(OMP) -o jacobi jacobi.cu

clean:
        rm -f val_test01_solved val_test02_solved *.o;
        rm -f omp_solved2 omp_solved3 omp_solved4 omp_solved5 omp_solved6;
        rm -f jacobi2D-omp gs2D-omp;