# homework 3

## OMP SCAN

benchmark with different threads


|    Threads   | Sequential | Parallel |
| ----------- | ----------- | ----------- |
| 1      | 0.414558s       |   0.637962s    |
| 2   | 0.400608s        | 0.546453s  |
| 3   | 0.393964s        | 0.240287s  |
| 4   | 0.397146s        | 0.205412s  |
| 5   | 0.395476s        |  0.189222s |
| 6   | 0.390131s        |  0.168190s |
| 7   | 0.393331s        |  0.239960s |
| 8   | 0.392442s        | 0.252725s  |

We can see that parallel code works best with 8 threads

Architecture
```text
Architecture:  x86_64
Processor Name: 6-Core Intel Core i7
Processor Speed: 2.2 GHz 
Number of processors: 1 
Total number of cores: 6 
L2 cache (per core): 256 KB 
L3 cache: 9 MB 
Hyper-Threading Technology: Enabled    
Memory: 16 GB 
```
