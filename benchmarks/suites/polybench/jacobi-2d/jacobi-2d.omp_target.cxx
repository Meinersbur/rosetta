// BUILD: add_benchmark(ppm=omp_target)

#include <rosetta.h>



static void kernel(pbsize_t tsteps, pbsize_t n,
                   multarray<real, 2> A, multarray<real, 2> B) {
    real *pA = &A[0,0];
    real *pB = &B[0,0];



#pragma omp target map(to:pA[0:n*n])  map(from:pB[0:n*n]) default(none) firstprivate(tsteps, n, A, B)
  {
#define AccA(x,y) (pA[(x)*n+(y)])
#define AccB(x,y) (pB[(x)*n+(y)])


    for (idx_t t = 0; t < tsteps; t++) {

#pragma omp target teams distribute parallel for collapse(2) dist_schedule(status) schedule(static)
      for (idx_t i = 1; i < n - 1; i++)
        for (idx_t j = 1; j < n - 1; j++)
          AccB(i,j) = (AccA(i,j) + AccA(i,j - 1) + AccA(i,1 + j) + AccA(1 + i,j) + AccA(i - 1,j)) / 5;

#pragma omp target teams distribute parallel for collapse(2) dist_schedule(status) schedule(static)
      for (idx_t i = 1; i < n - 1; i++)
        for (idx_t j = 1; j < n - 1; j++)
            AccA(i,j) = (AccB(i,j) + AccB(i,j - 1) + AccB(i,1 + j) + AccB(1 + i,j) + AccB(i - 1,j)) / 5;

    }
  }
}


void run(State &state, pbsize_t pbsize) {
  pbsize_t tsteps = 1; // 500
  pbsize_t n = pbsize; // 1300



  auto A = state.allocate_array<real>({n, n}, /*fakedata*/ true, /*verify*/ false, "A");
  auto B = state.allocate_array<real>({n, n}, /*fakedata*/ false, /*verify*/ true, "B");


  for (auto &&_ : state)
    kernel(tsteps, n, A, B);
}
