#include "rosetta.h"



static void kernel(int n, real *A) {
  for (int i = 0; i < n; i += 1)
    A[i] += 42;
}

void run(State &state, int n) {
  auto A = state.fakedata_array<real>(n, /*verify*/ true);

  for (auto &&_ : state)
    kernel(n, A.data());
}
