// BUILD: add_benchmark(ppm=omp_parallel,sources=[__file__, "gramschmidt-common.cxx"])

#include "gramschmidt-common.h"
#include <rosetta.h>




#if 0
void kernel_polly(pbsize_t m, pbsize_t n,
    real A[128][128], real  R[128][128], real  Q[128][128]) {
    __builtin_assume(m > 0);
    __builtin_assume(n > 0);
    //#pragma omp parallel default(none)
        {
            for (idx_t k = 0; k < n; k++) {

                R[k][k] = 0;
                for (idx_t i = 0; i < m; i++)
                    R[k][k] += A[i][k] * A[i][k];
                R[k][k] = R[k][k] * R[k][k]; // std::sqrt(R[k][k]);

                for (idx_t i = 0; i < m; i++)
                    Q[i][k] = A[i][k] / R[k][k];

                for (idx_t j = k + 1; j < n; j++) {

                    R[k][j] = 0;
                    for (idx_t i = 0; i < m; i++)
                        R[k][j] += Q[i][k] * A[i][j];

                    for (idx_t i = 0; i < m; i++)
                        A[i][j] -= Q[i][k] * R[k][j];
                }
            }
        }
}
#endif


static real sqr(real v) { return v * v; }

static void kernel(pbsize_t m, pbsize_t n,
                   multarray<real, 2> A, multarray<real, 2> R, multarray<real, 2> Q) {
  for (idx_t k = 0; k < n; k++) {
    real sum = 0;
    // FIXME: For some reason OpenMP-reduction numerically destabilizes this
    // Possibly inherent to Gram-Schmidt numeric instability
    // https://en.wikipedia.org/wiki/Gram–Schmidt_process#Numerical_stability
    // Generate fakedata that is not that similar to each other
#pragma omp parallel for schedule(static) default(none) firstprivate(k, m, A) \
                     reduction(+: sum)
    for (idx_t i = 0; i < m; i++) {
      //#pragma omp critical
      // printf("%lu %d: sqr(%g) = %g\n",k,i,A[i][k],sqr(A[i][k]) );
      sum += sqr(A[i][k]);
    }

    //    printf("%lu: sum=%g\n",k,sum );
    R[k][k] = std::sqrt(sum);


#pragma omp parallel for schedule(static) default(none) firstprivate(k, m, A, Q, R)
    for (idx_t i = 0; i < m; i++)
      Q[i][k] = A[i][k] / R[k][k];


#pragma omp parallel for schedule(static) default(none) firstprivate(k, m, n, A, Q, R)
    for (idx_t j = k + 1; j < n; j++) {
      R[k][j] = 0;
      for (idx_t i = 0; i < m; i++)
        R[k][j] += Q[i][k] * A[i][j];
    }

#pragma omp parallel for schedule(static) default(none) firstprivate(k, m, n, A, Q, R)
    for (idx_t j = k + 1; j < n; j++)
      for (idx_t i = 0; i < m; i++)
        A[i][j] -= Q[i][k] * R[k][j];
  }
}



void run(State &state, pbsize_t pbsize) {
    pbsize_t m = pbsize;              // 1200
    pbsize_t n = pbsize - pbsize / 6; // 1000
  

  auto A = state.allocate_array<real>({m, n}, /*fakedata*/ true, /*verify*/ true, "A");
  auto R = state.allocate_array<real>({n, n}, /*fakedata*/ false, /*verify*/ true, "R");
  auto Q = state.allocate_array<real>({m, n}, /*fakedata*/ false, /*verify*/ true, "Q");


  for (auto &&_ : state.manual()) {
    condition(m, n, A);
    {
      auto &&scope = _.scope();
      kernel(m, n, A, R, Q);
    }
  }
}
