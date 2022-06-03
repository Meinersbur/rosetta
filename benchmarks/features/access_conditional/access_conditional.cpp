#include "rosetta.h"

static void kernel(int n, double *A) {
    for (int i = 0; i < n; i += 1) {
        if ((i*(unsigned)i) % 3u)
          A[i] = i;
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
