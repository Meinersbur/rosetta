// BUILD: add_benchmark(ppm=omp_parallel)

#include "rosetta.h"


static void kernel(int m, int n,
                   multarray<real, 2> data,
                   multarray<real, 2> cov,
                   real *mean) {
#pragma omp parallel  default(none) firstprivate(m,n,data,cov,mean)
                       {
#pragma omp for schedule (static)
                           for (int j = 0; j < m; j++) {
                               mean[j] = 0.0;
                               for (int i = 0; i < n; i++)
                                   mean[j] += data[i][j];
                               mean[j] /= n;
                           }

#pragma omp for collapse(2) schedule (static)
                           for (int i = 0; i < n; i++)
                               for (int j = 0; j < m; j++)
                                   data[i][j] -= mean[j];

#pragma omp for collapse(2) // schedule (static)
                           for (int i = 0; i < m; i++)
                               for (int j = i; j < m; j++) {
                                   cov[i][j] = 0.0;
                                   for (int k = 0; k < n; k++)
                                       cov[i][j] += data[k][i] * data[k][j];
                                   cov[i][j] /= (n - 1.0);
                                   cov[j][i] = cov[i][j];
                               }
                       }
}


void run(State &state, int pbsize) {
  // n is 8%-25% larger than m
  size_t n = pbsize;
  size_t m = pbsize - pbsize / 8;

  auto data = state.allocate_array<real>({ n, m}, /*fakedata*/ true, /*verify*/ false);
  auto cov = state.allocate_array<real>({m, m}, /*fakedata*/ false, /*verify*/ true);
  auto mean = state.allocate_array<real>({m}, /*fakedata*/ false, /*verify*/ true);

  for (auto &&_ : state)
    kernel(m, n,  data, cov, mean);
}
