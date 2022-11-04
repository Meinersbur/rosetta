// BUILD: add_benchmark(ppm=omp_parallel)

#include <rosetta.h>
#include <omp.h>


static void kernel(pbsize_t m, pbsize_t n,
                   multarray<real, 2> data,
                   multarray<real, 2> corr,
                   real mean[],
                   real stddev[]) {
  

#pragma omp parallel default(none) firstprivate(m,n,data,corr,mean,stddev)
  {
            real eps = 0.1;

#pragma omp for schedule(static)
      for (idx_t j = 0; j < m; j++) {
          mean[j] = 0.0;
          for (int i = 0; i < n; i++)
              mean[j] += data[i][j];
          mean[j] /= n;
      }

#pragma omp for schedule(static)
      for (idx_t j = 0; j < m; j++) {
          stddev[j] = 0.0;
          for (idx_t i = 0; i < n; i++)
              stddev[j] += (data[i][j] - mean[j]) * (data[i][j] - mean[j]);
          stddev[j] /= n;
          stddev[j] = std::sqrt(stddev[j]);
          /* The following in an inelegant but usual way to handle
             near-zero std. dev. values, which below would cause a zero-
             divide. */
          stddev[j] = stddev[j] <= eps ? 1.0 : stddev[j];
      }


      /* Center and reduce the column vectors. */
#pragma omp for collapse(2)  schedule(static) 
      for (idx_t i = 0; i < n; i++)
          for (idx_t j = 0; j < m; j++) {
              data[i][j] -= mean[j];
              data[i][j] /= std::sqrt((real)n) * stddev[j];
          }


      /* Calculate the m * m correlation matrix. */
#pragma omp for 
      for (idx_t i = 0; i < m - 1; i++) {
          corr[i][i] = 1.0;
          for (int j = i + 1; j < m; j++) {
              corr[i][j] = 0.0;
              for (int k = 0; k < n; k++)
                  corr[i][j] += (data[k][i] * data[k][j]);
              corr[j][i] = corr[i][j];
          }
      }

#pragma omp single 
      {
      corr[m - 1][m - 1] = 1.0;
      }
  }
}


void run(State &state, int pbsize) {
  pbsize_t n = pbsize;
  pbsize_t m = pbsize - pbsize / 6;

  real float_n = n;
  auto data = state.allocate_array<real>({ n,m}, /*fakedata*/ true, /*verify*/ false);
  auto corr = state.allocate_array<real>({m, m}, /*fakedata*/ false, /*verify*/ true);
  auto mean = state.allocate_array<real>({m}, /*fakedata*/ false, /*verify*/ true);
  auto stddev = state.allocate_array<real>({m}, /*fakedata*/ false, /*verify*/ true);


  for (auto &&_ : state)
    kernel(m, n,  data, corr, mean, stddev);
}
