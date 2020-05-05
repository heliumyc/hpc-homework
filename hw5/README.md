# Usage

```$xslt
make clean
make
mpiexec -np [np] ./jacobi [-nl] [-max_iteration]
mpiexec -np [np] ./ssort [-N]

## or use bash script in slurm_scripts
cd slurm_scripts
bash runsort.sh
bash runweak.sh
bash runstrong.sh
```

# Jacobi reports

## Weak scaling

Max Iteration = 20000

|  N   | p   | N_l |  Time  | Residual  |
| ---- | --- | --- |  ----- | --------- |
| 100  | 1  | 100  | 0.212875 | 0.00513211 |
| 200  | 4  | 100  | 0.292883 | 14.1556 |
| 400  | 16 | 100  | 2.315778 | 176.08 |
| 800  | 64 | 100  | 2.025046 | 415.783 |
| 1600 | 256 | 100 | 2.570946 | 1375.32 |

## Strong scaling

Max Iteration = 20000

|  N   | p   | N_l |  Time  | Residual  |
| ---- | --- | --- |  ----- | --------- |
| 1600  | 1  | 1600  |  |  |
| 1600  | 4  | 800  | 74.024117 | 1375.32 |
| 1600  | 16 | 400  | 12.619656 | 1375.32 |
| 1600  | 64 | 200  | 5.305895 | 1375.32 |
| 1600 | 256 | 100  | 2.396963  |  1375.32 |

# Ssort reports
|  N   | Time  | 
| ---- | ----------- |
| 1e4  | 0.074737    |
| 1e5  | 0.485884    |
| 1e6  | 0.168195    |
