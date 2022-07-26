#include "rosetta.h"



static void kernel(int n,
                   multarray<real, 2> A) {
#pragma scop
  for (int i = 0; i < n; i++) {
    // j<i
    for (int j = 0; j < i; j++) {
      for (int k = 0; k < j; k++) {
        A[i][j] -= A[i][k] * A[j][k];
      }
      A[i][j] /= A[j][j];
    }
    // i==j case
    for (int k = 0; k < i; k++) {
      A[i][i] -= A[i][k] * A[i][k];
    }
    A[i][i] = std::sqrt(A[i][i]);
  }
#pragma endscop
}


void run(State &state, int pbsize) {
  size_t n = pbsize; // 2000



  auto A = state.allocate_array<real>({n, n}, /*fakedata*/ true, /*verify*/ true);



  for (auto &&_ : state)
    kernel(n, A);
}