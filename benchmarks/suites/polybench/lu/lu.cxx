// BUILD: add_benchmark(ppm=serial)

#include "rosetta.h"



static void kernel(int n,
                   multarray<real, 2> A) {
#pragma scop
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < i; j++) {
      for (int k = 0; k < j; k++)
        A[i][j] -= A[i][k] * A[k][j];
        auto val = A[j][j];
        if (val == 0)
      std::cerr << "Div by zero";
      A[i][j] /= val;
    }
    for (int j = i; j < n; j++) {
      for (int k = 0; k < i; k++)
        A[i][j] -= A[i][k] * A[k][j];
    }
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


  auto A = state.allocate_array<real>({n, n}, /*fakedata*/ true, /*verify*/ true, "A");



    for (auto&& _ : state.manual()) {
        ensure_fullrank(n, A);
        {
            auto &&scope = _.scope();
            kernel(n, A);
        }
    }
}
