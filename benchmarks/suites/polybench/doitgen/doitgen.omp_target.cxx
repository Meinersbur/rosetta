// BUILD: add_benchmark(ppm=omp_target)

#include <rosetta.h>


static void kernel(pbsize_t nr, pbsize_t nq, pbsize_t np,
    multarray<real, 3> A, multarray<real, 2> C4, multarray<real, 3> sum) {
    real *pA = &A[0][0][0];
    real *pC4 = &C4[0][0];
    real *psum = &sum[0][0][0];

#pragma omp target teams distribute parallel for collapse(2)  map(tofrom:pA[0:nr*nq*np])  map(to:pC4[0:np*np]) map(alloc:psum[0:nr*nq*np])
            for (idx_t r = 0; r < nr; r++)
                for (idx_t q = 0; q < nq; q++) {
                    for (idx_t p = 0; p < np; p++) {
                        psum[(r*nq+q)*np+p] = 0;
                        for (idx_t s = 0; s < np; s++)
                            psum[(r*nq+q)*np+p] += pA[(r*nq+q)*np+s] * pC4[s*np+p];
                    }
                    for (idx_t p = 0; p < np; p++)
                        pA[(r*nq+q)*np+p] = psum[(r*nq+q)*np+p];
                }

}





void run(State &state, pbsize_t pbsize) {
  pbsize_t nq = pbsize - pbsize / 8;  // 140
  pbsize_t nr = pbsize - pbsize / 16; // 150
  pbsize_t np = pbsize;               // 160


  auto A = state.allocate_array<real>({nr, nq, np}, /*fakedata*/ true, /*verify*/ true, "A");
  auto C4 = state.allocate_array<real>({np, np}, /*fakedata*/ true, /*verify*/ false, "C4");
  auto sum = state.allocate_array<real>({nr, nq, np}, /*fakedata*/ false, /*verify*/ false, "sum");

  for (auto &&_ : state)
    kernel(nr, nq, np, A, C4,sum );
}

