// BUILD: add_benchmark(ppm=omp_target)

#include <rosetta.h>



static void kernel(pbsize_t tsteps, pbsize_t n,
                   multarray<real, 2> A) {
  real *pA = &A[0][0];

#pragma omp target data map(tofrom \
                            : pA [0:n * n])
  {
#define AccA(x, y) (pA[(x)*n + (y)])


    for (idx_t t = 0; t <= tsteps - 1; t++) {


      // FIXME: Parallelizing this should give different results
#pragma omp target teams distribute parallel for collapse(2) schedule(static) default(none) firstprivate(tsteps, n, pA)
      for (idx_t i = 1; i <= n - 2; i++)
        for (idx_t j = 1; j <= n - 2; j++)
          AccA(i, j) = (AccA(i - 1, j - 1) + AccA(i - 1, j) + AccA(i - 1, j + 1) + AccA(i, j - 1) + AccA(i, j) + AccA(i, j + 1) + AccA(i + 1, j - 1) + AccA(i + 1, j) + AccA(i + 1, j + 1)) / 9;
    }
  }
}


void run(State &state, int pbsize) {
  pbsize_t tsteps = 1; // 500
  pbsize_t n = pbsize; // 2000



  auto A = state.allocate_array<real>({n, n}, /*fakedata*/ true, /*verify*/ false, "A");



  for (auto &&_ : state)
    kernel(tsteps, n, A);
}
