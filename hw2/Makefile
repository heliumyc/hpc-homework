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
	$(CC) $(CFLAGS) $(STD) $(OMP) -march=native -o MMult1 MMult1.cpp

ompbugs:
	$(CC) $(CFLAGS) $(STD) $(OMP) -o omp_solved2 omp_bug2_solved.c;
	$(CC) $(CFLAGS) $(STD) $(OMP) -o omp_solved3 omp_bug3_solved.c;
	$(CC) $(CFLAGS) $(STD) $(OMP) -o omp_solved4 omp_bug4_solved.c;
	$(CC) $(CFLAGS) $(STD) $(OMP) -o omp_solved5 omp_bug5_solved.c;
	$(CC) $(CFLAGS) $(STD) $(OMP) -o omp_solved6 omp_bug6_solved.c;

jacobi:
	$(CC) $(CFLAGS) $(STD) $(OMP) -march=native -o jacobi2D-omp jacobi2D-omp.cpp;

gs:
	$(CC) $(CFLAGS) $(STD) $(OMP) -march=native -o gs2D-omp gs2D-omp.cpp;

clean:
	rm -f val_test01_solved val_test02_solved *.o;
	rm -f omp_solved2 omp_solved3 omp_solved4 omp_solved5 omp_solved6;
	rm -f jacobi2D-omp gs2D-omp;

test:
	./val_test01_solved;
	./val_test02_solved;
	./omp_bug2_solved;
	./omp_bug3_solved;
	./omp_bug4_solved;
	./omp_bug5_solved;
	./omp_bug6_solved;
	./jacobi2D-omp 100 -1 6;
	./gs2D-omp 100 -1 6;
