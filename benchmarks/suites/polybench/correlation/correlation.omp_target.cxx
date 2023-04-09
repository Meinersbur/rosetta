// BUILD: add_benchmark(ppm=omp_target)

#include <rosetta.h>
#include <omp.h>



static void kernel(pbsize_t m, pbsize_t n,
                   multarray<real, 2> data,
                   multarray<real, 2> corr,
                   real mean[],
                   real stddev[]) {
    real *pdata = &data[0][0];
    real *pcorr = &corr[0][0];
    real eps = 0.1;
     
#pragma omp target data map(to:pdata[0:n*m]) map(from:mean[0:m],stddev[0:m],pcorr[0:m*m]) map(to:n,m, eps)
  {


#pragma omp target teams distribute parallel for default(none) dist_schedule(static) schedule(static) firstprivate(m,n,mean,pdata)
    for (idx_t j = 0; j < m; j++) {
      mean[j] = 0.0;
      for (idx_t i = 0; i < n; i++)
        mean[j] += pdata[i*m+j];
      mean[j] /= n;
    }

#pragma omp target teams distribute parallel for dist_schedule(static) schedule(static) default(none) firstprivate(m,n,stddev,pdata,mean,eps)
    for (idx_t j = 0; j < m; j++) {
      stddev[j] = 0.0;
      for (idx_t i = 0; i < n; i++)
        stddev[j] += (pdata[i*m+j] - mean[j]) * (pdata[i*m+j] - mean[j]);
      stddev[j] /= n;
      stddev[j] = std::sqrt(stddev[j]);
      /* The following in an inelegant but usual way to handle
         near-zero std. dev. values, which below would cause a zero-
         divide. */
      if (stddev[j] <= eps)
        stddev[j] = 1.0;
    }


    /* Center and reduce the column vectors. */
#pragma omp target teams distribute parallel for collapse(2) dist_schedule(static) schedule(static)  default(none) firstprivate(n,m,pdata,mean,stddev)
    for (idx_t i = 0; i < n; i++)
      for (idx_t j = 0; j < m; j++) {
        pdata[i*m+j] -= mean[j];
        pdata[i*m+j] /= std::sqrt((real)n) * stddev[j];
      }


      /* Calculate the m * m correlation matrix. */
#pragma omp target teams distribute parallel for dist_schedule(static) schedule(static) default(none) firstprivate(m,n,pcorr,pdata)
    for (idx_t i = 0; i < m - 1; i++) {
      pcorr[i*m+i] = 1.0;
      for (idx_t j = i + 1; j < m; j++) {
        pcorr[i*m+j] = 0.0;
        for (idx_t k = 0; k < n; k++)
          pcorr[i*m+j] += (pdata[k*m+i] * pdata[k*m+j]);
        pcorr[j*m+i] = pcorr[i*m+j];
      }
    }

#pragma omp target 
      pcorr[(m - 1)*m+(m - 1)] = 1.0;
  }
}


void run(State &state, pbsize_t pbsize) {
  pbsize_t n = pbsize;
  pbsize_t m = pbsize - pbsize / 6;


  auto data = state.allocate_array<real>({n, m}, /*fakedata*/ true, /*verify*/ false, "data");
  auto mean = state.allocate_array<real>({m}, /*fakedata*/ false, /*verify*/ true, "mean");
  auto stddev = state.allocate_array<real>({m}, /*fakedata*/ false, /*verify*/ true, "stddev");
  auto corr = state.allocate_array<real>({m, m}, /*fakedata*/ false, /*verify*/ true, "corr");



  for (auto &&_ : state)
    kernel(m, n, data, corr, mean, stddev);
}
