// BUILD: add_benchmark(ppm=omp_target)

#include <rosetta.h>


static void kernel(pbsize_t n,
                   multarray<real, 2> L, real x[], real b[]) {
    real *Ldata = &L[0][0];
#pragma omp target data map(to:Ldata[0:n*n],b[0:n]) map(from:x[0:n]) 
                       {


                           for (idx_t i = 0; i < n; i++) {
                               real sum = 0;
#pragma omp target teams distribute parallel for \
             reduction(+ : sum)
                               for (idx_t j = 0; j < i; j++)
                                   sum += L[i][j] * x[j];
#pragma omp target
                               x[i] = (b[i] - sum) / L[i][i];
                           }


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
