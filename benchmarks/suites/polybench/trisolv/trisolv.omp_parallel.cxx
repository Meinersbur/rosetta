// BUILD: add_benchmark(ppm=omp_parallel)

#include <rosetta.h>


static void kernel(pbsize_t n,
                   multarray<real, 2> L, real x[], real b[]) {
  for (idx_t i = 0; i < n; i++) {
    real sum = 0;
#pragma omp parallel for schedule(static) default(none) \
             firstprivate(i, x, L) \
             reduction(+ : sum)
    for (idx_t j = 0; j < i; j++)
      sum += L[i][j] * x[j];
     //   sum +=  L[i][j];
   x[i] = (b[i] - sum) / L[i][i];
   // x[i] =  1 +sum;
  }
}


void run(State &state, pbsize_t pbsize) {
  pbsize_t n = pbsize; // 2000


  auto L = state.allocate_array<real>({n, n}, /*fakedata*/ true, /*verify*/ false, "L");
  auto x = state.allocate_array<real>({n}, /*fakedata*/ false, /*verify*/ true, "x");
  auto b = state.allocate_array<real>({n}, /*fakedata*/ true, /*verify*/ false, "b");

  for (auto &&_ : state)
    kernel(n, L, x, b);
}
