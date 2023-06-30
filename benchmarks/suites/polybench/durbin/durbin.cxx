// BUILD: add_benchmark(ppm=serial,sources=[__file__, "durbin-common.cxx"])

#include <rosetta.h>
#include "durbin-common.h"


static void kernel(pbsize_t n,
                   real r[],
                   real y[], real z[]) {
#pragma scop
  y[0] = -r[0];
  real beta = 1;
  real alpha = -r[0];

  for (idx_t k = 1; k < n; k++) {
    beta = (1 - alpha * alpha) * beta;
    real sum = 0;
    for (idx_t i = 0; i < k; i++)
      sum += r[k - i - 1] * y[i];

    alpha = -(r[k] + sum) / beta;

    for (idx_t i = 0; i < k; i++)
      z[i] = y[i] + alpha * y[k - i - 1];

    for (idx_t i = 0; i < k; i++)
      y[i] = z[i];

    y[k] = alpha;
  }
#pragma endscop
}


void run(State &state, pbsize_t pbsize) {
  pbsize_t n = pbsize; // 2000

  auto r = state.allocate_array<real>({n}, /*fakedata*/ false, /*verify*/ false, "r");
  auto y = state.allocate_array<real>({n}, /*fakedata*/ false, /*verify*/ true, "y");
  auto z = state.allocate_array<real>({n}, /*fakedata*/ false, /*verify*/ false, "z");

  for (auto &&_ : state.manual()) {
    initialize_input_vector(n, r);
    {
      auto &&scope = _.scope();
      kernel(n, r, y, z);
    }
  }
}
