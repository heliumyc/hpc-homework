CFLAGS:=-O3  # -g = debug, -O2 for optimized code CPPFLAGS for g++
CC:=g++
OMP:=-fopenmp
STD:=-std=c++11

.PHONY: all

all: val01 val02 mmult ompbugs jacobi gs;

val01:
	$(CC) $(CFLAGS) $(STD) -o val_test01_solved val_test01_solved.cpp

val02:
	$(CC) $(CFLAGS) $(STD)  -o val_test02_solved val_test02_solved.cpp

mmult:
	$(CC) $(CFLAGS) $(STD) $(OMP) MMult1 MMult1.cpp

ompbugs:
	$(CC) $(CFLAGS) $(STD) $(OMP) -o omp_solved2 omp_solved2.cpp;
	$(CC) $(CFLAGS) $(STD) $(OMP) -o omp_solved3 omp_solved3.cpp;
	$(CC) $(CFLAGS) $(STD) $(OMP) -o omp_solved4 omp_solved4.cpp;
	$(CC) $(CFLAGS) $(STD) $(OMP) -o omp_solved5 omp_solved5.cpp;
	$(CC) $(CFLAGS) $(STD) $(OMP) -o omp_solved6 omp_solved6.cpp;

jacobi:
	$(CC) $(CFLAGS) $(STD) $(OMP)  -o jacobi2D-omp jacobi2D-omp.cpp;

gs:
	$(CC) $(CFLAGS) $(STD) $(OMP)  -o gs2D-omp gs2D-omp.cpp;

clean:
	rm -f val_test01_solved val_test02_solved *.o;
	rm -f omp_solved2 omp_solved3 omp_solved4 omp_solved5 omp_solved6;
	rm -f jacobi2D-omp gs2D-omp;
