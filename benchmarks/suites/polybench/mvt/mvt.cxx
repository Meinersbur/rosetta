#include "rosetta.h"



static void kernel(int n,
                   real x1[],
real x2[],
real y_1[],
real y_2[],
      multarray<real, 2> A) {
#pragma scop
  for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++)
      x1[i] = x1[i] + A[i][j] * y_1[j];
  for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++)
      x2[i] = x2[i] + A[j][i] * y_2[j];
#pragma endscop
}


void run(State &state, int pbsize) {
  size_t n = pbsize;



    auto A = state.allocate_array<real>({n, n}, /*fakedata*/ true, /*verify*/ false);
  auto x1 = state.allocate_array<real>({n, }, /*fakedata*/ true, /*verify*/ true);
  auto x2 = state.allocate_array<real>({n}, /*fakedata*/ false, /*verify*/ true);
  auto y_1 = state.allocate_array<real>({n}, /*fakedata*/ true, /*verify*/ false);
  auto y_2 = state.allocate_array<real>({n}, /*fakedata*/ true, /*verify*/ false);


  for (auto &&_ : state)
    kernel(n, x1,x2,y_1,y_2,A);
}
