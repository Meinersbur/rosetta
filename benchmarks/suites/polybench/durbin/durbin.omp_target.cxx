// BUILD: add_benchmark(ppm=omp_target,sources=[__file__, "durbin-common.cxx"])

#include "durbin-common.h"
#include <rosetta.h>



static void kernel(pbsize_t n,
                   real r[],
                   real y[], real z[]) {
  y[0] = -r[0];
  real beta = 1;
  real alpha = -r[0];


#pragma omp target data map(tofrom                                              \
                            : y [0:n]) map(to                                   \
                                           : r [0:n]) map(alloc                 \
                                                          : z [0:n]) map(tofrom \
                                                                         : alpha, beta)
  {


    for (idx_t k = 1; k < n; k++) {
      real sum = 0;

      {
#pragma omp target teams distribute parallel for schedule(static) default(none) firstprivate(k, r, y) reduction(+ \
                                                                                                                : sum)
        for (idx_t i = 0; i < k; i++)
          sum += r[k - i - 1] * y[i];

#pragma omp target map(tofrom \
                       : alpha, beta) // FIXME: libomp seems to get reference counting from without the map clause
        {
          beta = (1 - alpha * alpha) * beta;
          alpha = -(r[k] + sum) / beta;
        }
      }

#pragma omp target update from(alpha, beta) // FIXME: alpha,beta should stay on the device, why doesn't it?


#pragma omp target teams distribute parallel for schedule(static) default(none) firstprivate(k, z, y, alpha, y)
      for (idx_t i = 0; i < k; i++)
        z[i] = y[i] + alpha * y[k - i - 1];

#pragma omp target teams distribute parallel for schedule(static) default(none) firstprivate(k, y, z)
      for (idx_t i = 0; i < k; i++)
        y[i] = z[i];

#pragma omp target
      y[k] = alpha;
    }
  }
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
