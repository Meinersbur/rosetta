#include "cholesky-common.h"

// https://math.stackexchange.com/a/358092
// https://math.stackexchange.com/q/357980
void ensure_posdefinite(int n, multarray<real, 2> A) {
#if 1
  if (n == 3) {
    A[0][0] = 4;
    A[0][1] = 12;
    A[0][2] = -16;
    A[1][0] = 12;
    A[1][1] = 37;
    A[1][2] = -43;
    A[2][0] = -16;
    A[2][1] = -43;
    A[2][2] = 98;
    return;
  } else if (n == 4) {
    real B[4][4] = {0};
    B[0][0] = 1;
    B[1][0] = 2;
    B[2][0] = 3;
    B[3][0] = 4;
    B[1][1] = 5;
    B[2][1] = 6;
    B[3][1] = 7;
    B[2][2] = 8;
    B[3][2] = 9;
    B[3][3] = 10;

    for (idx_t i = 0; i < n; ++i)
      for (idx_t j = 0; j < n; ++j)
        A[i][j] = 0;

    for (idx_t i = 0; i < n; ++i)
      for (idx_t j = 0; j < n; ++j)
        for (idx_t k = 0; k < n; ++k)
          A[i][j] += B[i][k] * B[j][k];
    return;
  } else if (n == 5) {
    real B[5][5] = {0};
    int k = 1;
    for (idx_t i = 0; i < n; ++i)
      for (idx_t j = 0; j <= i; ++j) {
        B[i][j] = k;
        k += 1;
      }



    for (idx_t i = 0; i < n; ++i)
      for (idx_t j = 0; j < n; ++j)
        A[i][j] = 0;

    for (idx_t i = 0; i < n; ++i)
      for (idx_t j = 0; j < n; ++j)
        for (idx_t k = 0; k < n; ++k)
          A[i][j] += B[i][k] * B[j][k];
    return;
  } else if (n == 6) {
    real B[6][6] = {0};
    int k = 1;
    for (idx_t i = 0; i < n; ++i)
      for (idx_t j = 0; j < n; ++j)
        if (j >= i) {
          B[j][i] = k;
          k += 1;
        }



    for (idx_t i = 0; i < n; ++i)
      for (idx_t j = 0; j < n; ++j)
        A[i][j] = 0;

    for (idx_t i = 0; i < n; ++i)
      for (idx_t j = 0; j < n; ++j)
        for (idx_t k = 0; k < n; ++k)
          A[i][j] += B[i][k] * B[j][k];
    return;
  } else if (n == 7) {
    real B[7][7] = {0};
    int k = 2;
    for (idx_t i = 0; i < n; ++i)
      for (idx_t j = 0; j < n; ++j)
        if (j >= i) {
          B[j][i] = k;
          k += 1;
        }



    for (idx_t i = 0; i < n; ++i)
      for (idx_t j = 0; j < n; ++j)
        A[i][j] = 0;

    for (idx_t i = 0; i < n; ++i)
      for (idx_t j = 0; j < n; ++j)
        for (idx_t k = 0; k < n; ++k)
          A[i][j] += B[i][k] * B[j][k];
    return;
  }
#endif


  // make symmetric (not really necessary, the kernel doesn't read the upper triangular elements anyway)
  for (int i = 0; i < n; i++)
    for (int j = 0; j < i; j++) {
      auto val = (std::abs(A[i][j]) + std::abs(A[j][i])) / 2;
      A[i][j] = val;
      A[j][i] = val;
    }



  real maximum = 0;
  for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++) {
      auto val = std::abs(A[i][j]);
      if (val > maximum)
        maximum = val;
    }


  // Make the diagnonal elements too large to be a linear combination of the other columns (strictly diagonally dominant).
  for (int i = 0; i < n; i++)
    A[i][i] = std::abs(A[i][i]) + 1 + n * maximum;


  // FIXME: Repeated invocation will grow the numbers
}
