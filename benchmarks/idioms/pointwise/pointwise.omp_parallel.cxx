#include "rosetta.h"



static void kernel(int n, double *A) {
#pragma omp parallel for
  for (int i = 0; i < n; i += 1)
    A[i] += 42;
}

void run(State &state, int n) {
  auto A = state.fakedata_array<double>(n, /*verify*/ true);

  // TODO: #pragma omp parallel outside of loop
  for (auto &&_ : state)
    kernel(n, A.data());
}
