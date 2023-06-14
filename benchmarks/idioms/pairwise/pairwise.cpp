#include "rosetta.h"
#include "pairwise.h"



static void kernel(int n, double *C, double *B, double *A) {
    for (int i = 0; i < n; i += 1)
        for (int j = 0; j < n; j += 1)
            C[i * n + j] = A[i] * B[j];
}

void run(benchmark::State& state, int n) {
    double *A = new double[n];
    double *B = new double[n];
    double *C = new double[n*n];

    for (auto &&_ : state) {
        kernel(n, C, B, A);
        benchmark::ClobberMemory();
    }

    delete[] A;
    delete[] B;
    delete[] C;
}
