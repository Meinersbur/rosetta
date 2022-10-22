// BUILD: add_benchmark(ppm=serial)
#include "rosetta.h"



static void kernel(int n,
                   multarray<real, 2> A) {
#pragma omp parallel for schedule(static)

  for (int i = 0; i < n; i++) {
    // j<i
    for (int j = 0; j < i; j++) {
      for (int k = 0; k < j; k++) 
        A[i][j] -= A[i][k] * A[j][k];
      A[i][j] /= A[j][j]; 
    }
    // i==j case
    for (int k = 0; k < i; k++) 
      A[i][i] -= A[i][k] * A[i][k];
    


    A[i][i] = std::sqrt( A[i][i]);
  }
}

// https://math.stackexchange.com/a/358092
static void ensure_posdefinite(int n, multarray<real, 2> A) {
    // make symmetric (not really necessary, the kernel doesn't read the upper triangular elements anyway)
    for (int i = 0; i < n; i++)
        for (int j = 0; j < i; j++) {
            auto val = (std::abs(A[i][j]) + std::abs(A[j][i]))/2;
            A[i][j] = val;
            A[j][i] = val;
        }

        

    real maximum=0;
    for (int i = 0; i < n ; i++)
        for (int j = 0; j < n; j++) {
            auto val = std::abs(A[i][j]);
            if (val > maximum) maximum = val;
        }


    // Make the diagnonal elements too large to be a linear combination of the other columns (strictly diagonally dominant).
    for (int i = 0; i < n ; i++)
        A[i][i] = std::abs( A[i][i]) + 1 + n*maximum;


    // FIXME: Repeated invocation will grow the numbers
}


void run(State &state, int pbsize) {
  size_t n = pbsize; // 2000

  auto A = state.allocate_array<real>({n, n}, /*fakedata*/ true, /*verify*/ true, "A");



  for (auto&& _ : state.manual()) {
      ensure_posdefinite(n, A);
      {
          auto &&scope = _.scope();
          // FIXME: cholesky of pos-definite matrix is not necessarily itself pos-definite
          kernel(n, A);
      }
  }
}
