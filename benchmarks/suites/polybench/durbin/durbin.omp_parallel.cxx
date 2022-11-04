// BUILD: add_benchmark(ppm=omp_parallel)

#include "rosetta.h"



static void kernel(pbsize_t n,
                   real r[],
                   real y[], real z[]) {
    y[0] = -r[0];
    real beta = 1;
    real alpha = -r[0];

    for (idx_t k = 1; k < n; k++) {
        real sum = 0;
#pragma omp parallel for default(none) reduction(+:sum) firstprivate(k,r,y) schedule(static)
        for (idx_t i = 0; i < k; i++) 
            sum += r[k - i - 1] * y[i];
        

        beta = (1 - alpha * alpha) * beta;
        alpha = -(r[k] + sum) / beta;

#pragma omp parallel 
        {
#pragma omp for schedule(static)
            for (idx_t i = 0; i < k; i++)
                z[i] = y[i] + alpha * y[k - i - 1];

#pragma omp for schedule(static)
            for (idx_t i = 0; i < k; i++)
                y[i] = z[i];
        }
        
        y[k] = alpha;
    }
}


void run(State &state, pbsize_t pbsize) {
    pbsize_t n = pbsize; // 2000



  auto r = state.allocate_array<real>({n}, /*fakedata*/ true, /*verify*/ false);
  auto y = state.allocate_array<real>({n}, /*fakedata*/ false, /*verify*/ true);
  auto z = state.allocate_array<real>({n}, /*fakedata*/ false, /*verify*/ false);

  for (auto &&_ : state)
    kernel(n, r, y, z);
}