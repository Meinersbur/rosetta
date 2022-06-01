#include "rosetta.h"



static void kernel(int n, double *A) {
    for (int i = 0; i < n; i += 1)
        A[i] += 42;
}

void run(State& state, int n) {
    // default size
    if (n < 0)
        n = (DEFAULT_N);

    double *A = state.malloc<double>(n);


    for (auto &&_ : state) {
            kernel(n, A);
    }

    state.free(A);
}



