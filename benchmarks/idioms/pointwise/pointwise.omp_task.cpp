#include "rosetta.h"


static void kernel(int n, double *B, double *A) {
    #pragma omp taskloop
    for (int i = 0; i < n; i += 1) {
        B[i] = 42 * A[i];
    }
    #pragma omp taskwait
}


 void run(State& state, int n) {
    double *A = new double[n];
    double *B = new double[n];

    for (auto &&_ : state) {
        kernel(n, B, A);
        benchmark::ClobberMemory();
    }

    delete[] A;
    delete[] B;
}



