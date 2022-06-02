#include "rosetta.h"



static void kernel(int n, double *A) {
    #pragma omp parallel for
    for (int i = 0; i < n; i += 1)
        A[i] += 42;
}

void run(State& state, int n) {
    // default size
    if (n < 0)
        n = (DEFAULT_N);

    double *A = state.malloc<double>(n);
    state.fakedata(A, n);

    for (auto &&_ : state) {
        kernel(n, A);
        state.verifydata(A, n);
    }

    state.free(A);
}


