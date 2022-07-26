#include "rosetta.h"



static void kernel(int m, int n,
                   multarray<real, 2> A, multarray<real, 2> R, multarray<real, 2> Q) {
#pragma scop
  for (int k = 0; k < n; k++) {
    real nrm = 0;
    for (int i = 0; i < m; i++)
      nrm += A[i][k] * A[i][k];
    R[k][k] = std::sqrt(nrm);
    for (int i = 0; i < m; i++)
      Q[i][k] = A[i][k] / R[k][k];
    for (int j = k + 1; j < n; j++) {
      R[k][j] = 0;
      for (int i = 0; i < m; i++)
        R[k][j] += Q[i][k] * A[i][j];
      for (int i = 0; i < m; i++)
        A[i][j] = A[i][j] - Q[i][k] * R[k][j];
    }
  }
#pragma endscop
}


void run(State &state, int pbsize) {
  size_t m = pbsize - pbsize / 6; // 1000
  size_t n = pbsize;              // 1200


  auto A = state.allocate_array<real>({m, n}, /*fakedata*/ true, /*verify*/ true);
  auto R = state.allocate_array<real>({n, n}, /*fakedata*/ true, /*verify*/ true);
  auto Q = state.allocate_array<real>({m, n}, /*fakedata*/ true, /*verify*/ true);

  for (auto &&_ : state)
    kernel(m, n, A, R, Q);
}
