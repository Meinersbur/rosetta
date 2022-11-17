// BUILD: add_benchmark(ppm=omp_parallel)

#include <rosetta.h>



static void kernel(pbsize_t n, pbsize_t m,
                   real alpha,
                   multarray<real, 2> B, multarray<real, 2> A) {
#pragma omp parallel default(none) firstprivate(n,m,alpha,B,A)
                       {
#pragma omp for 
                           for (idx_t j = 0; j < n; j++) 
                                for (idx_t i = 0; i < m; i++)                             
                                   for (idx_t k = i + 1; k < m; k++)
                                       B[i][j] += A[k][i] * B[k][j];

#pragma omp for collapse(2) schedule(static)
                                   for (idx_t i = 0; i < m; i++)
                                       for (idx_t j = 0; j < n; j++)
                                   B[i][j] *= alpha;
                               
                       }
}



void run(State &state, pbsize_t pbsize) {
    pbsize_t n = pbsize;
    pbsize_t m = pbsize - pbsize / 6;

  real alpha = 1.5;
  auto B = state.allocate_array<real>({m, n}, /*fakedata*/ true, /*verify*/ true, "B");
  auto A = state.allocate_array<real>({n, m}, /*fakedata*/ true, /*verify*/ false, "A");

  for (auto &&_ : state)
    kernel(n, m, alpha, B, A);
}
