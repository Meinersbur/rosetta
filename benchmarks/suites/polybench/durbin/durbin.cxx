#include "rosetta.h"



static void kernel(int n,
                   real r[],
                   real y[], real z[]) {
#pragma scop
  y[0] = -r[0];
  real beta = 1;
  real alpha = -r[0];

  for (int k = 1; k < n; k++) {
    beta = (1 - alpha * alpha) * beta;
    real sum = 0;
    for (int i = 0; i < k; i++) {
      sum += r[k - i - 1] * y[i];
    }
    alpha = -(r[k] + sum) / beta;

    for (int i = 0; i < k; i++) {
      z[i] = y[i] + alpha * y[k - i - 1];
    }
    for (int i = 0; i < k; i++) {
      y[i] = z[i];
    }
    y[k] = alpha;
  }
#pragma endscop
}


void run(State &state, int pbsize) {
  size_t n = pbsize; // 2000



  auto r = state.allocate_array<real>({n}, /*fakedata*/ true, /*verify*/ false);
  auto y = state.allocate_array<real>({n}, /*fakedata*/ false, /*verify*/ true);
  auto z = state.allocate_array<real>({n}, /*fakedata*/ false, /*verify*/ false);

  for (auto &&_ : state)
    kernel(n, r, y, z);
}