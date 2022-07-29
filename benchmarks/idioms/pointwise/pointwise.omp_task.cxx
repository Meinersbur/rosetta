#include "rosetta.h"



static void kernel(int n, real A[]) {
#pragma omp taskloop
  for (int i = 0; i < n; i += 1)
    A[i] += 42;
}

void run(State &state, int n) {
    auto A = state.allocate_array<real>({n}, /*fakedata*/true, /*verify*/true, "A");


  for (auto &&_ : state) {
    kernel(n,A);
#pragma omp taskwait // TODO: do in rosetta-omp_task
  }
}
