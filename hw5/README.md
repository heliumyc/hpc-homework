# Usage

```$xslt
make clean
make
mpiexec -np [np] ./jacobi [-nl] [-max_iteration]
mpiexec -np [np] ./ssort [-N]

## or use bash script in slurm_scripts
cd slurm_scripts
bash runsort.sh # submit sort jobs to slurm
bash runweak.sh  # submit weak scaling jobs to slurm
bash runstrong.sh  # submit strong scaling jobs to slurm
```

# Jacobi reports

## Weak scaling

Max Iteration = 20000

|  N   | p   | N_l |  Time  | Residual  |
| ---- | --- | --- |  ----- | --------- |
| 100  | 1  | 100  | 0.271625 | 0.00513211 |
| 200  | 4  | 100  | 0.436735 | 14.1556 |
| 400  | 16 | 100  | 2.723155 | 176.08 |
| 800  | 64 | 100  | 4.073693 | 575.316 |
| 1600 | 256 | 100 | 2.729070 | 1375.32 |

## Strong scaling

Max Iteration = 20000

|  N   | p   | N_l |  Time  | Residual  |
| ---- | --- | --- |  ----- | --------- |
| 1600  | 1  | 1600 | 557.565293 | 1375.32 |
| 1600  | 4  | 800  | 120.552866 | 1375.32 |
| 1600  | 16 | 400  | 14.503465 | 1375.32 |
| 1600  | 64 | 200  | 7.600227 | 1375.32 |
| 1600 | 256 | 100  | 2.520968  |  1375.32 |

# Ssort reports

Partition: c01_17

Cores: 64

|  N   | Time  | 
| ---- | ----------- |
| 1e4  | 0.074081    |
| 1e5  | 0.069223    |
| 1e6  | 0.188538    |
