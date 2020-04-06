#include <algorithm>
#include <stdio.h>
#include <math.h>
#include <omp.h>

// Scan A array and write result into prefix_sum array;
// use long data type to avoid overflow
void scan_seq(long* prefix_sum, const long* A, long n) {
  if (n == 0) return;
  prefix_sum[0] = 0;
  for (long i = 1; i < n; i++) {
    prefix_sum[i] = prefix_sum[i-1] + A[i-1];
  }
}

void scan_omp(long* prefix_sum, const long* A, long n) {
  // TODO: implement multi-threaded OpenMP scan
  int p = 6;
  long len = (n-1)/p + ((n-1)%p > 0);
  long* partial_sum = (long*) malloc(p* sizeof(long));
  std::fill(partial_sum, partial_sum+p, 0);
  prefix_sum[0] = 0;
  omp_set_num_threads(p);

#pragma omp parallel for default(shared)
    {
        for (int k = 0; k < p; k++) {
            prefix_sum[len * k + 1] = A[len * k];
            int i = len * k + 2;
            for (; i < n && i <= len * (k + 1); i++) {
                prefix_sum[i] = prefix_sum[i - 1] + A[i - 1];
            }
            partial_sum[k] = prefix_sum[i - 1];
        }
    }

#pragma omp single
            for (int k = 1; k < p; k++) {
                partial_sum[k] += partial_sum[k - 1];
            }

#pragma omp parallel for default(shared)
    {
        for (int k = 1; k < p; k++) {
            for (long i = len * k + 1; i < n && i <= len * (k + 1); i++) {
                prefix_sum[i] += partial_sum[k - 1];
            }
        }
    }

    free(partial_sum);

}

int main() {
  long N = 100000000;
  long* A = (long*) malloc(N * sizeof(long));
  long* B0 = (long*) malloc(N * sizeof(long));
  long* B1 = (long*) malloc(N * sizeof(long));
    for (long i = 0; i < N; i++) A[i] = rand();

  double tt = omp_get_wtime();
  scan_seq(B0, A, N);
  printf("sequential-scan = %fs\n", omp_get_wtime() - tt);

  tt = omp_get_wtime();
  scan_omp(B1, A, N);
  printf("parallel-scan   = %fs\n", omp_get_wtime() - tt);

  long err = 0;
  for (long i = 0; i < N; i++) err = std::max(err, std::abs(B0[i] - B1[i]));
  printf("error = %ld\n", err);

  free(A);
  free(B0);
  free(B1);
  return 0;
}
