#include "rosetta.h"



static void kernel(int n,
                   multarray<real, 2> A, real b[], real x[], real y[]) {
#pragma scop
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < i; j++) {
      real w = A[i][j];
      for (int k = 0; k < j; k++) 
        w -= A[i][k] * A[k][j];
      A[i][j] = w / A[j][j];
    }
    for (int j = i; j < n; j++) {
      real w = A[i][j];
      for (int k = 0; k < i; k++) 
        w -= A[i][k] * A[k][j];
      A[i][j] = w;
    }
  }

  for (int i = 0; i < n; i++) {
    real w = b[i];
    for (int j = 0; j < i; j++)
      w -= A[i][j] * y[j];
    y[i] = w;
  }

  for (int i = n - 1; i >= 0; i--) {
    real w = y[i];
    for (int j = i + 1; j < n; j++)
      w -= A[i][j] * x[j];
    x[i] = w / A[i][i];
  }
#pragma endscop
}



static void ensure_fullrank(int n,  multarray<real, 2> A) {
    real maximum=0;
    for (int i = 0; i < n ; i++)
        for (int j = 0; j < n; j++) {
            auto val = std::abs(A[i][j]);
            if (val > maximum) maximum = val;
        }

    // Make the diagnonal elements too large to be a linear combination of the other columns without also making the other vector elements too large.
    for (int i = 0; i < n ; i++)
        A[i][i] = std::abs( A[i][i]) + 1 + maximum;
}


void run(State &state, int pbsize) {
  size_t n = pbsize; // 2000


   
  auto A = state.allocate_array<real>({n, n}, /*fakedata*/ true, /*verify*/ false, "A");
  auto b = state.allocate_array<real>({n}, /*fakedata*/ true, /*verify*/ false, "b");
  auto x = state.allocate_array<real>({n}, /*fakedata*/ true, /*verify*/ true, "x");
  auto y = state.allocate_array<real>({n}, /*fakedata*/ true, /*verify*/ false, "y");



  for (auto&& _ : state.manual()) {
      ensure_fullrank(n, A);
      {
          auto &&scope = _.scope();
          kernel(n, A, b, x, y);
      }
  }
}
