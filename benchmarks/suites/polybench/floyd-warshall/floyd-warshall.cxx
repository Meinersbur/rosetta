#include "rosetta.h"


static void kernel(int n, multarray<real, 2> path) {
#pragma scop
  for (int k = 0; k < n; k++) {
    for (int i = 0; i < n; i++)
      for (int j = 0; j < n; j++)
        path[i][j] = path[i][j] < path[i][k] + path[k][j] ? path[i][j] : path[i][k] + path[k][j];
  }
#pragma endscop
}



void run(State &state, int pbsize) {
  size_t n = pbsize; // 2800



  auto path = state.allocate_array<double>({n, n}, /*fakedata*/ true, /*verify*/ true);



  for (auto &&_ : state)
    kernel(n, path);
}
