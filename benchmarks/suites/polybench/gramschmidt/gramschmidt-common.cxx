#include "gramschmidt-common.h"



void condition(pbsize_t m, pbsize_t n, multarray<real, 2> A) {
  for (idx_t i = 0; i < m; i++) {
    for (idx_t j = 0; j < n; j++) {
#if 1
      if (std::abs((int)i - j) > 1)
        continue;

      A[i][j] = 0.25;
#else
      A[i][j] = (i == j) ? 1 : 0;
#endif
    }
  }
}
