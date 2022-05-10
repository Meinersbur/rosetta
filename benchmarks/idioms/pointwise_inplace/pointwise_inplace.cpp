#include "pointwise_inplace.h"
#include <benchmark/benchmark.h>
#include <cstdlib>
#include <string>

static void kernel(int n, double *A) {
    for (int i = 0; i < n; i += 1) {
        A[i] += 42;
    }
}


 void run(benchmark::State& state, int n) {
    double *A = new double[n];

    for (auto &&_ : state) {
        kernel(n, A);
        benchmark::ClobberMemory();
    }

    delete[] A;
}





