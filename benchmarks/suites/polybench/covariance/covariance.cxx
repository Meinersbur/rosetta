// BUILD: add_benchmark(ppm=serial)

#include "rosetta.h"


static void kernel(pbsize_t m, pbsize_t n,
                   multarray<real, 2> data,
                   multarray<real, 2> cov,
                   real mean[]) {
#pragma scop
  for (idx_t j = 0; j < m; j++) {
    mean[j] = 0.0;
    for (idx_t i = 0; i < n; i++)
      mean[j] += data[i][j];
    mean[j] /= n;
  }

  for (idx_t i = 0; i < n; i++)
    for (idx_t j = 0; j < m; j++)
      data[i][j] -= mean[j];

  for (idx_t i = 0; i < m; i++)
    for (idx_t j = i; j < m; j++) {
      cov[i][j] = 0.0;
      for (idx_t k = 0; k < n; k++)
        cov[i][j] += data[k][i] * data[k][j];
      cov[i][j] /= (n - 1.0);
      cov[j][i] = cov[i][j];
    }
#pragma endscop
}


void run(State &state, pbsize_t pbsize) {
  // n is 8%-25% larger than m
  pbsize_t n = pbsize;
  pbsize_t m = pbsize - pbsize / 8;

  auto data = state.allocate_array<real>({ n, m}, /*fakedata*/ true, /*verify*/ false, "data");
    auto mean = state.allocate_array<real>({m}, /*fakedata*/ false, /*verify*/ true, "mean");
  auto cov = state.allocate_array<real>({m, m}, /*fakedata*/ false, /*verify*/ true, "cov");



  for (auto &&_ : state)
    kernel(m, n,  data, cov, mean);
}
