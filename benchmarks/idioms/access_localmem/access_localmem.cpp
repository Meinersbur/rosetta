#include "rosetta.h"

#define LOCAL_SIZE 32

static void kernel(int n, double *A, double Local[LOCAL_SIZE]) {
    for (int i = 0; i < n; i += 1)
        A[i] = Local[(3*i) % LOCAL_SIZE];
}

void run(benchmark::State& state, int n) {
    if (n < 0)
      n = (DEFAULT_N);

    double *A = new double[LOCAL_SIZE];

    // Supposed to fit into local (CUDA: shared) memory
    // TODO: also write to it so we cannot use constant memory
    double Local[LOCAL_SIZE];

    for (auto &&_ : state) {
        kernel(n, A, Local);
        benchmark::ClobberMemory();
    }

    delete[] B;
}
