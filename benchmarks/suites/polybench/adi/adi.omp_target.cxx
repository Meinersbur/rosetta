// BUILD: add_benchmark(ppm=omp_target)

#include <rosetta.h>


static void kernel(
    pbsize_t tsteps,
    pbsize_t n,
    multarray<real, 2> u,
    multarray<real, 2> v,
    multarray<real, 2> p,
    multarray<real, 2> q) {
  real *udata = &u[0][0];
  real *vdata = &v[0][0];
  real *pdata = &p[0][0];
  real *qdata = &q[0][0];

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


#pragma omp target data map(tofrom                       \
                            : udata [0:n * n]) map(alloc \
                                                   : vdata [0:n * n], pdata [0:n * n], qdata [0:n * n])
  {

    for (idx_t t = 1; t <= tsteps; t++) {

// Column Sweep
#pragma omp target teams distribute parallel for
      for (idx_t i = 1; i < n - 1; i++) {
        vdata[0 * n + i] = 1;
        pdata[i * n + 0] = 0;
        qdata[i * n + 0] = vdata[0 * n + i];
        for (idx_t j = 1; j < n - 1; j++) {
          pdata[i * n + j] = -c / (a * pdata[i * n + (j - 1)] + b);
          qdata[i * n + j] = (-d * udata[j * n + (i - 1)] + (1 + 2 * d) * udata[j * n + i] - f * udata[j * n + (i + 1)] - a * qdata[i * n + (j - 1)]) / (a * pdata[i * n + (j - 1)] + b);
        }

        vdata[(n - 1) * n + i] = 1;
        for (idx_t j = n - 2; j >= 1; j--)
          vdata[j * n + i] = pdata[i * n + j] * vdata[(j + 1) * n + i] + qdata[i * n + j];
      }

// Row Sweep
#pragma omp target teams distribute parallel for
      for (idx_t i = 1; i < n - 1; i++) {
        udata[i * n + 0] = 1;
        pdata[i * n + 0] = 0;
        qdata[i * n + 0] = udata[i * n + 0];
        for (idx_t j = 1; j < n - 1; j++) {
          pdata[i * n + j] = -f / (d * pdata[i * n + (j - 1)] + e);
          qdata[i * n + j] = (-a * vdata[(i - 1) * n + j] + (1 + 2 * a) * vdata[i * n + j] - c * vdata[(i + 1) * n + j] - d * qdata[i * n + (j - 1)]) / (d * pdata[i * n + (j - 1)] + e);
        }
        udata[i * n + (n - 1)] = 1;
        for (idx_t j = n - 2; j >= 1; j--)
          udata[i * n + j] = pdata[i * n + j] * udata[i * n + (j + 1)] + qdata[i * n + j];
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
