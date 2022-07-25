#include "rosetta.h"



static void kernel(int n,
  multarray<real, 2> A) {
#pragma scop
  for (int i = 0; i < n; i++) {
    for (int j = 0; j <i; j++) {
       for (int k = 0; k < j; k++) {
          A[i][j] -= A[i][k] * A[k][j];
       }
        A[i][j] /= A[j][j];
    }
   for (int j = i; j < n; j++) {
       for (int k = 0; k < i; k++) {
          A[i][j] -= A[i][k] * A[k][j];
       }
    }
  }
#pragma endscop
}


void run(State &state, int pbsize) {
size_t n = pbsize; // 2000


    auto A = state.allocate_array<real>({n,n}, /*fakedata*/ true, /*verify*/ true);


  for (auto &&_ : state)
    kernel(n,A);
}
