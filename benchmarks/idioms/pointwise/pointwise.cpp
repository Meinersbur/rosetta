#include "rosetta.h"

static void kernel(int n, double *B, double *A) {
    for (int i = 0; i < n; i += 1)
        B[i] = 42 + A[i];
}

void run(benchmark::State& state, int n) {
    // default size
    if (n < 0)
        n = (DEFAULT_N);

    double *A = new double[n];
    double *B = new double[n];

    for (auto &&_ : state) {
        kernel(n, B, A);
        benchmark::ClobberMemory();
    }

    delete[] A;
    delete[] B;
}
