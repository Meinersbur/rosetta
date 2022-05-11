#include "rosetta.h"
#include <cmath>

static void kernel(int n, double *A) {
    for (int i = 0; i < n; i += 1)
      A[i] = sqrt(i);
}

void run(benchmark::State& state, int n) {
    // default size
    if (n < 0)
        n = (DEFAULT_N);

    double *A = new double[n];

    for (auto &&_ : state) {
        kernel(n, A);
        benchmark::ClobberMemory();
    }

    delete[] A;
}
