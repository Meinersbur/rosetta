// BUILD: add_benchmark(ppm=omp_task)
#include "rosetta.h"



static void kernel(pbsize_t n, real A[]) {
  #pragma omp parallel 
  {
    #pragma omp single 
    {
#pragma omp taskloop
  for (idx_t i = 0; i < n; i += 1)
    A[i] += 42;
    #pragma omp taskwait // TODO: do in rosetta-omp_task
  }
  }
}

void run(State &state, pbsize_t n) {
    auto A = state.allocate_array<real>({n}, /*fakedata*/true, /*verify*/true, "A");


  for (auto &&_ : state) {
    kernel(n,A);
  }
}
