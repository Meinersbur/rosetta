#include "rosetta.h"

// piecewise reduction
static void kernel(int n, double *A, double *B, double *C) {
    for (int i = 0; i < n; i += 1) {
      for (int j = 0; j < n; j += 1)
         A[i] += B[i] * C[j];
    }
}

void run(benchmark::State& state, int n) {
    // default size
    if (n < 0)
        n = (DEFAULT_N);

    double *A = new double[n];
    double *B = new double[n];
    double *C = new double[n];

    for (auto &&_ : state) {
        kernel(n, A, B, C);
        benchmark::ClobberMemory();
    }

    delete[] A;
    delete[] B;
    delete[] C;
}
