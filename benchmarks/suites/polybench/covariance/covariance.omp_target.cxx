// BUILD: add_benchmark(ppm=omp_target)

#include <rosetta.h>


static void kernel(pbsize_t m, pbsize_t n,
                   multarray<real, 2> data,
                   multarray<real, 2> cov,
                   real *mean) {
  real *pdata = &data[0][0];
  real *pcov = &cov[0][0];

#pragma omp target data map(to                          \
                            : pdata [0:n * m]) map(from \
                                                   : pcov [0:m * m], mean [0:m])
  {
#define Accdata(i, j) (pdata[(i)*m + (j)])
#define Acccov(i, j) (pcov[(i)*m + (j)])

#pragma omp target teams distribute parallel for dist_schedule(static) schedule(static) default(none) firstprivate(m, n, mean, pdata)
    for (idx_t j = 0; j < m; j++) {
      mean[j] = 0.0;
      for (idx_t i = 0; i < n; i++)
        mean[j] += Accdata(i, j);
      mean[j] /= n;
    }

#pragma omp target teams distribute parallel for collapse(2) dist_schedule(static) schedule(static) default(none) firstprivate(m, n, pdata, mean)
    for (idx_t i = 0; i < n; i++)
      for (idx_t j = 0; j < m; j++)
        Accdata(i, j) -= mean[j];

#pragma omp target teams distribute parallel for collapse(2) default(none) firstprivate(m, n, pcov, pdata)
    for (idx_t i = 0; i < m; i++)
      for (idx_t j = i; j < m; j++) {
        Acccov(i, j) = 0.0;
        for (idx_t k = 0; k < n; k++)
          Acccov(i, j) += Accdata(k, i) * Accdata(k, j);
        Acccov(i, j) /= (n - 1.0);
        Acccov(j, i) = Acccov(i, j);
      }

  } // #pragma omp target data
}



void run(State &state, pbsize_t pbsize) {
  // n is 8%-25% larger than m
  pbsize_t n = pbsize;
  pbsize_t m = pbsize - pbsize / 8;

  auto data = state.allocate_array<real>({n, m}, /*fakedata*/ true, /*verify*/ false, "data");
  auto mean = state.allocate_array<real>({m}, /*fakedata*/ false, /*verify*/ true, "mean");
  auto cov = state.allocate_array<real>({m, m}, /*fakedata*/ false, /*verify*/ true, "cov");

  for (auto &&_ : state)
    kernel(m, n, data, cov, mean);
}
