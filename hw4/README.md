# Build

```asm
// make all
make
// delete executable and out file
make clean
```

# Benchmarks

for some reason, program won't stop at cuda server 4

perhaps it is related to double precision?

- inner product

dimension: 1<<25

GB/s

|  cims gpu   | Sequential  | Openmp    | Cuda |
| ----------- | ----------- | ----------- | ----------- |
| 1      | 3.957516  |   26.457461    |  50.729626  |
| 2     |    3.888401     | 26.636471 | 95.646915   |
| 3     |     3.551317    |20.765977  | 75.581704  |
| 4     |         |  |   |
| 5     |   3.148826      | 15.788291 | 41.319464  |

- matrix vector multiplication

dimension: 1<<13

|  cims gpu   | Sequential  | Openmp    | Cuda |
| ----------- | ----------- | ----------- | ----------- |
| 1      | 13.785757  |   59.838451    |  84.677590  |
| 2     |   13.871154      |54.447479  |  314.850515  |
| 3     |    10.041173     | 24.649074 | 131.154740  |
| 4     |         |  |   |
| 5     |   13.623630      | 16.809243 | 69.977220  |


# Usage of Jacobi

```
./jacobi {N} {max iteration}
```

if max iteration is -1 , then it will not stop until it converges

like
```
./jacobi 100 -1
./jacobi 200 -1
```

this works on cuda server 1, but gpu version
 is as quick as openmp version. 
 
 spent days trying to optimize it but failed
