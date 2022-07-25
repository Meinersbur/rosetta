#include "rosetta.h"


static void kernel(int n,
  multarray<real, 2> L, real x[], real b[]) {
#pragma scop
  for (int i = 0; i < n; i++)
    {
      x[i] = b[i];
      for (int j = 0; j <i; j++)
        x[i] -= L[i][j] * x[j];
      x[i] = x[i] / L[i][i];
    }
#pragma endscop
}


void run(State &state, int pbsize) {
size_t n = pbsize; // 2000


    auto L = state.allocate_array<real>({n,n}, /*fakedata*/ true, /*verify*/ false);
    auto x = state.allocate_array<real>({n}, /*fakedata*/ true, /*verify*/ true);
    auto b = state.allocate_array<real>({n}, /*fakedata*/ true, /*verify*/ false);

  for (auto &&_ : state)
    kernel(n,L,x,b);
}
