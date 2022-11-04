// BUILD: add_benchmark(ppm=omp_parallel)

#include "rosetta.h"


static void kernel(int nr, int nq, int np,
                   multarray<real, 3> A,
                   multarray<real, 2> C4) {
#pragma omp parallel default(none) firstprivate(nr,nq,np, A, C4)
                       {
#pragma omp for collapse(2)schedule(static) 
                           for (int r = 0; r < nr; r++)
                               for (int q = 0; q < nq; q++)                         for (int p = 0; p < np; p++)                      {
                                       A[r][q][p] = 0;
                                       for (int s = 0; s < np; s++)
                                           A[r][q][p] += A[r][q][s] * C4[s][p];
                                   }    
                       }
}


void run(State &state, int pbsize) {
  size_t nq = pbsize - pbsize / 8;  // 140
  size_t nr = pbsize - pbsize / 16; // 150
  size_t np = pbsize;               // 160


  auto A = state.allocate_array<double>({nr, nq, np}, /*fakedata*/ true, /*verify*/ true, "A");
  auto C4 = state.allocate_array<double>({np, np}, /*fakedata*/ true, /*verify*/ false, "C4");


  for (auto &&_ : state)
    kernel(nr, nq, np, A, C4);
}
