#include "rosetta.h"



static void kernel(int tsteps, int n,
                   multarray<real, 2> A, multarray<real, 2> B) {
#pragma scop
  for (int t = 0; t < tsteps; t++) {
    for (int i = 1; i < n - 1; i++)
      for (int j = 1; j < n - 1; j++)
        B[i][j] = ((real)0.2) * (A[i][j] + A[i][j - 1] + A[i][1 + j] + A[1 + i][j] + A[i - 1][j]);
    for (int i = 1; i < n - 1; i++)
      for (int j = 1; j < n - 1; j++)
        A[i][j] = ((real)0.2) * (B[i][j] + B[i][j - 1] + B[i][1 + j] + B[1 + i][j] + B[i - 1][j]);
  }
#pragma endscop
}


void run(State &state, int pbsize) {
  size_t tsteps = 1; // 500
  size_t n = pbsize; // 1300



  auto A = state.allocate_array<real>({n, n}, /*fakedata*/ true, /*verify*/ true);
  auto B = state.allocate_array<real>({n, n}, /*fakedata*/ true, /*verify*/ false);


  for (auto &&_ : state)
    kernel(tsteps, n, A, B);
}
