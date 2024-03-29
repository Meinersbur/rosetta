// BUILD: add_benchmark(ppm=omp_parallel)

#include <rosetta.h>


static void kernel(
    pbsize_t tsteps,
    pbsize_t n,
    multarray<real, 2> u,
    multarray<real, 2> v,
    multarray<real, 2> p,
    multarray<real, 2> q) {

  real DX = 1 / (real)n;
  real DY = 1 / (real)n;
  real DT = 1 / (real)tsteps;
  real B1 = 2;
  real B2 = 1;
  real mul1 = B1 * DT / (DX * DX);
  real mul2 = B2 * DT / (DY * DY);

  real a = -mul1 / 2;
  real b = 1 + mul1;
  real c = a;
  real d = -mul2 / 2;
  real e = 1 + mul2;
  real f = d;

#pragma omp parallel
  {
    for (idx_t t = 1; t <= tsteps; t++) {
// Column Sweep
#pragma omp for schedule(static)
      for (idx_t i = 1; i < n - 1; i++) {
        v[0][i] = 1;
        p[i][0] = 0;
        q[i][0] = v[0][i];
        for (idx_t j = 1; j < n - 1; j++) {
          p[i][j] = -c / (a * p[i][j - 1] + b);
          q[i][j] = (-d * u[j][i - 1] + (1 + 2 * d) * u[j][i] - f * u[j][i + 1] - a * q[i][j - 1]) / (a * p[i][j - 1] + b);
        }

        v[n - 1][i] = 1;
        for (idx_t j = n - 2; j >= 1; j--)
          v[j][i] = p[i][j] * v[j + 1][i] + q[i][j];
      }
// Row Sweep
#pragma omp for schedule(static)
      for (idx_t i = 1; i < n - 1; i++) {
        u[i][0] = 1;
        p[i][0] = 0;
        q[i][0] = u[i][0];
        for (idx_t j = 1; j < n - 1; j++) {
          p[i][j] = -f / (d * p[i][j - 1] + e);
          q[i][j] = (-a * v[i - 1][j] + (1 + 2 * a) * v[i][j] - c * v[i + 1][j] - d * q[i][j - 1]) / (d * p[i][j - 1] + e);
        }
        u[i][n - 1] = 1;
        for (idx_t j = n - 2; j >= 1; j--)
          u[i][j] = p[i][j] * u[i][j + 1] + q[i][j];
      }
    }
  }
}



void run(State &state, pbsize_t pbsize) {
  pbsize_t tsteps = pbsize / 2; // 500
  pbsize_t n = pbsize;          // 1000



  auto u = state.allocate_array<double>({n, n}, /*fakedata*/ true, /*verify*/ true, "u");
  auto v = state.allocate_array<double>({n, n}, /*fakedata*/ false, /*verify*/ false, "v");
  auto p = state.allocate_array<double>({n, n}, /*fakedata*/ false, /*verify*/ false, "p");
  auto q = state.allocate_array<double>({n, n}, /*fakedata*/ false, /*verify*/ false, "q");



  for (auto &&_ : state)
    kernel(tsteps, n, u, v, p, q);
}
