#include "rosetta.h"



static void kernel(int n, double *A) {
#pragma omp taskloop
  for (int i = 0; i < n; i += 1)
    A[i] += 42;
}

void run(State &state, int n) {
  auto A = state.fakedata_array<double>(n, /*verify*/ true);

  for (auto &&_ : state) {
    kernel(n, A.data());
#pragma omp taskwait // TODO: do in rosetta-omp_task
  }
}
