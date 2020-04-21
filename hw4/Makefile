CFLAGS:=-O3  # -g = debug, -O2 for optimized code CPPFLAGS for g++
CC:=nvcc
OMP:=-Xcompiler -fopenmp
STD:=-std=c++11

.PHONY: all

all:  mulvec innerproduct jacobi

mulvec:
	$(CC) $(CFLAGS) $(STD) $(OMP) -o vec_mat_mul vec_mat_mul.cu

innerproduct:
	$(CC) $(CFLAGS) $(STD) $(OMP) -o inner_product inner_product.cu

jacobi:
	$(CC) $(CFLAGS) $(STD) $(OMP) -o jacobi jacobi.cu

clean:
	rm -f vec_mat_mul inner_product jacobi;
