// BUILD: add_benchmark(ppm=serial)

#include "rosetta.h"



static void kernel(pbsize_t tmax,
                   pbsize_t nx,
                   pbsize_t ny,
                   multarray<real, 2> ex, multarray<real, 2> ey, multarray<real, 2> hz, real fict[]) {
#pragma scop
  for (idx_t t = 0; t < tmax; t++) {
    for (idx_t j = 0; j < ny; j++)
      ey[0][j] = fict[t];
    for (idx_t i = 1; i < nx; i++)
      for (idx_t j = 0; j < ny; j++)
        ey[i][j] -= (real)(0.5) * (hz[i][j] - hz[i - 1][j]);
    for (idx_t i = 0; i < nx; i++)
      for (idx_t j = 1; j < ny; j++)
        ex[i][j] -= (real)(0.5) * (hz[i][j] - hz[i][j - 1]);
    for (idx_t i = 0; i < nx - 1; i++)
      for (idx_t j = 0; j < ny - 1; j++)
        hz[i][j] -= (real)(0.7) * (ex[i][j + 1] - ex[i][j] + ey[i + 1][j] - ey[i][j]);
  }
#pragma endscop
}


void run(State &state, int pbsize) {
  pbsize_t tmax = 5 * pbsize / 12;   // 500
  pbsize_t nx = pbsize - pbsize / 6; // 1000
  pbsize_t ny = pbsize;              // 1200



  auto ex = state.allocate_array<real>({nx, ny}, /*fakedata*/ true, /*verify*/ true, "ex");
  auto ey = state.allocate_array<real>({nx, ny}, /*fakedata*/ true, /*verify*/ true, "ey");
  auto hz = state.allocate_array<real>({nx, ny}, /*fakedata*/ true, /*verify*/ true, "hz");
  auto fict = state.allocate_array<real>({tmax}, /*fakedata*/ true, /*verify*/ false, "fict");



  for (auto &&_ : state)
    kernel(tmax, nx, ny, ex, ey, hz, fict);
}
