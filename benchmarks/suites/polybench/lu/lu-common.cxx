#include "lu-common.h"

void ensure_fullrank(pbsize_t n, multarray<real, 2> A) {
  real maximum = 0;
  for (idx_t i = 0; i < n; i++)
    for (idx_t j = 0; j < n; j++) {
      auto val = std::abs(A[i][j]);
      if (val > maximum)
        maximum = val;
    }

  // Make the diagnonal elements too large to be a linear combination of the other columns without also making the other vector elements too large.
  for (idx_t i = 0; i < n; i++)
    A[i][i] = std::abs(A[i][i]) + 1 + maximum;
}
