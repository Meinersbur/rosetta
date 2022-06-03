#include "rosetta.h"

static void kernel(int n, double *A, int *Index) {
    for (int i = 0; i < n; i += 1)
        A[Index[i]] += 1;
}

void run(State& state, int n) {
    double *A = new double[n];
    int *Index = new int[n];

    for (int i = 0; i < n; i += 1)
        Index[i] = (i * 7) % n;

    for (auto &&_ : state) {
        kernel(n, A, Index);
        ClobberMemory();
    }

    delete[] A;
    delete[] Index;
}
