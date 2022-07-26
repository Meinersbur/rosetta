#include "rosetta.h"



static void kernel(int tsteps, int n,
                   real A[], real B[]) {
#pragma scop
  for (int t = 0; t < tsteps; t++) {
    for (int i = 1; i < n - 1; i++)
      B[i] = (real)0.33333 * (A[i - 1] + A[i] + A[i + 1]);
    for (int i = 1; i < n - 1; i++)
      A[i] = (real)0.33333 * (B[i - 1] + B[i] + B[i + 1]);
  }
#pragma endscop
}


void run(State &state, int pbsize) {
  size_t tsteps = 1; // 500
  size_t n = pbsize; // 2000



  auto A = state.allocate_array<real>({n}, /*fakedata*/ true, /*verify*/ true);
  auto B = state.allocate_array<real>({n}, /*fakedata*/ true, /*verify*/ false);


  for (auto &&_ : state)
    kernel(tsteps, n, A, B);
}
