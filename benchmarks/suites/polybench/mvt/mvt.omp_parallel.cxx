// BUILD: add_benchmark(ppm=omp_parallel)

#include <rosetta.h>



static void kernel(pbsize_t  n,
                   real x1[],
                   real x2[],
                   real y_1[],
                   real y_2[],
                   multarray<real, 2> A) {
#pragma omp parallel default(none) firstprivate(n,x1,x2,y_1,y_2,A)
                       {
#pragma omp for schedule(static) nowait
                           for (idx_t i = 0; i < n; i++)
                               for (idx_t j = 0; j < n; j++)
                                   x1[i] += A[i][j] * y_1[j];
#pragma omp for schedule(static)
                           for (idx_t i = 0; i < n; i++)
                               for (idx_t j = 0; j < n; j++)
                                   x2[i] += A[j][i] * y_2[j];
                       }
}


void run(State &state, pbsize_t pbsize) {
    pbsize_t n = pbsize; // 2000



  auto A = state.allocate_array<real>({n, n}, /*fakedata*/ true, /*verify*/ false, "A");
  auto x1 = state.allocate_array<real>({n},/*fakedata*/ true, /*verify*/ true, "x1");
  auto x2 = state.allocate_array<real>({n}, /*fakedata*/ true, /*verify*/ true, "x2");
  auto y_1 = state.allocate_array<real>({n}, /*fakedata*/ true, /*verify*/ false, "y_1");
  auto y_2 = state.allocate_array<real>({n}, /*fakedata*/ true, /*verify*/ false, "y_2");


  for (auto &&_ : state)
    kernel(n, x1, x2, y_1, y_2, A);
}
