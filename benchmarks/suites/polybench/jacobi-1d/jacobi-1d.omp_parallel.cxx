// BUILD: add_benchmark(ppm=omp_parallel)

#include "rosetta.h"



static void kernel(pbsize_t tsteps, pbsize_t n,  real A[], real B[]) {
#pragma omp parallel default(none) firstprivate(tsteps,n,A,B)
    {
        for (idx_t t = 0; t < tsteps; t++) {

#pragma omp for schedule(static)
            for (idx_t i = 1; i < n - 1; i++)
                B[i] = (A[i - 1] + A[i] + A[i + 1])/3;

#pragma omp for schedule(static)
            for (idx_t i = 1; i < n - 1; i++)
                A[i] =  (B[i - 1] + B[i] + B[i + 1])/3;

        }
    }
}


void run(State &state, pbsize_t pbsize) {
    pbsize_t tsteps = 1; // 500
    pbsize_t n = pbsize; // 2000



  auto A = state.allocate_array<real>({n}, /*fakedata*/ true, /*verify*/ true);
  auto B = state.allocate_array<real>({n}, /*fakedata*/ true, /*verify*/ false);


  for (auto &&_ : state)
    kernel(tsteps, n, A, B);
}
