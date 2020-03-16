# Homework 2

Detailed solution is in *latex/hw2.pdf*

## Build

```asm
// make all
make  
// delete executable and out file
make clean 
```

## Usage

For omp bug 4, must run
```
ulimit -s unlimited
```

For Jacobi and Gauss

```
./program {N} {max iteration} {number of thread}
```

if max iteration is -1 , then it will not stop until it converges

like
```
./jacobi2D-omp 100 -1 8 
./gs2D-ompg  200 -1 6
```
