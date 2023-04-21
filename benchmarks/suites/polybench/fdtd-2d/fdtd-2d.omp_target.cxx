// BUILD: add_benchmark(ppm=omp_target)

#include <rosetta.h>



static void kernel(pbsize_t tmax,
                   pbsize_t nx,
                   pbsize_t ny,
                   multarray<real, 2> ex, multarray<real, 2> ey, multarray<real, 2> hz, real fict[]) {
  real *pex = &ex[0][0];
  real *pey = &ey[0][0];
  real *phz = &hz[0][0];


#pragma omp target data map(tofrom                                                      \
                            : pex [0:nx * ny], pey [0:nx * ny], phz [0:nx * ny]) map(to \
                                                                                     : fict [0:tmax])
  {
    for (idx_t t = 0; t < tmax; t++) {

#pragma omp target teams distribute parallel for dist_schedule(static) schedule(static) // nowait
      for (idx_t j = 0; j < ny; j++)
        pey[0 * ny + j] = fict[t];

#pragma omp target teams distribute parallel for collapse(2) dist_schedule(static) schedule(static) // nowait (seems to cause crashes in libomp.so)
      for (idx_t i = 1; i < nx; i++)
        for (idx_t j = 0; j < ny; j++)
          pey[i * ny + j] -= (phz[i * ny + j] - phz[(i - 1) * ny + j]) / 2;

#pragma omp target teams distribute parallel for collapse(2) dist_schedule(static) schedule(static)
      for (idx_t i = 0; i < nx; i++)
        for (idx_t j = 1; j < ny; j++)
          pex[i * ny + j] -= (phz[i * ny + j] - phz[i * ny + (j - 1)]) / 2;

#pragma omp target teams distribute parallel for collapse(2) dist_schedule(static) schedule(static)
      for (idx_t i = 0; i < nx - 1; i++)
        for (idx_t j = 0; j < ny - 1; j++)
          //   phz[i*ny+j] = 42;
          phz[i * ny + j] -= (real)(0.7) * (pex[i * ny + (j + 1)] - pex[i * ny + j] + pey[(i + 1) * ny + j] - pey[i * ny + j]);
    }
  }
}


void run(State &state, pbsize_t pbsize) {
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
