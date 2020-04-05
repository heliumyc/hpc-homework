### Compilation:

```gcc main.c -lm -O3 -o out```

### Usage:

```./program N MaxIteration Method (1 -> Jacobi, 2 -> Gauss) [LOG-RESIDUAL]```

like

```./out 100 5000 1 1``` means N=100, max iteration = 5000 and use Jacobi
output logs to the screen

if MaxIteration is -1 then it will not stop until it converges

### Task in homework
- to compare N=100 and N=10000 iterations until convergence

    ```gcc main.c -lm -O3 -o out```
    
    ```./out 100 -1 1 0```
    
    ```./out 100 -1 2 0```
    
    ```./out 10000 -1 1 0```
    
    ```./out 10000 -1 2 0```

- to compare -O0 and -03 with N=10000 and iteration=100

    ```gcc main.c -c lm -O0 -o out0 ```
    
    ```gcc main.c -c lm -O3 -o out3 ```
    
    ```./out0 10000 100 1 0```
    
    ```./out3 10000 100 1 0```
    
    ```./out0 10000 100 2 0```
    
    ```./out3 10000 100 2 0```
      
