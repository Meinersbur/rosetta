// BUILD: add_benchmark(ppm=serial)

#include "rosetta.h"


static void kernel(pbsize_t  n, multarray<real, 2> path) {
#pragma scop
  for (idx_t k = 0; k < n; k++) {
    for (idx_t i = 0; i < n; i++)
      for (idx_t j = 0; j < n; j++)
        path[i][j] = path[i][j] < path[i][k] + path[k][j] ? path[i][j] : path[i][k] + path[k][j];
  }
#pragma endscop
}



void run(State &state, pbsize_t pbsize) {
    pbsize_t n = pbsize; // 2800



  auto path = state.allocate_array<real>({n, n}, /*fakedata*/ true, /*verify*/ true, "path");



  for (auto &&_ : state)
    kernel(n, path);
}
