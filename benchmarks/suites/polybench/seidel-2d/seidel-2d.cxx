#include "rosetta.h"



static void kernel(int tsteps, int n,
                   multarray<real, 2> A) {
#pragma scop
  for (int t = 0; t <= tsteps - 1; t++)
    for (int i = 1; i <= n - 2; i++)
      for (int j = 1; j <= n - 2; j++)
        A[i][j] = (A[i - 1][j - 1] + A[i - 1][j] + A[i - 1][j + 1] + A[i][j - 1] + A[i][j] + A[i][j + 1] + A[i + 1][j - 1] + A[i + 1][j] + A[i + 1][j + 1]) / 9;
#pragma endscop
}


void run(State &state, int pbsize) {
  size_t tsteps = 1; // 500
  size_t n = pbsize; // 2000



  auto A = state.allocate_array<real>({n, n}, /*fakedata*/ true, /*verify*/ true);



  for (auto &&_ : state)
    kernel(tsteps, n, A);
}
