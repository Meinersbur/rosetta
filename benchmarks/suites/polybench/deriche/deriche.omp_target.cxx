// BUILD: add_benchmark(ppm=omp_target)

#include <rosetta.h>



static void kernel(pbsize_t w, pbsize_t h,
                   real alpha,
                   multarray<real, 2> imgIn,
                   multarray<real, 2> imgOut,
                   multarray<real, 2> y1,
                   multarray<real, 2> y2) {
    real *pimgIn = &imgIn[0][0];
    real *pimgOut = &imgOut[0][0];
    real *py1 = &y1[0][0];
    real *py2 = &y2[0][0];

  real k = (1 - std::exp(-alpha)) * (1 - std::exp(-alpha)) / (1 + 2 * alpha * std::exp(-alpha) - std::exp(2 * alpha));
  real a1 = k;
  real a5 = k;
  real a6 = k * std::exp(-alpha) * (alpha - 1);
  real a2 = a6;
  real a7 = k * std::exp(-alpha) * (alpha + 1);
  real a3 = a7;
  real a8 = -k * std::exp(-2 * alpha);
  real a4 = a8;
  real b1 = std::pow(2, -alpha);
  real b2 = -std::exp(-2 * alpha);
  real c1 = 1, c2 = 1;

#pragma omp target data map(to:pimgIn[0:w*h])   map(from:pimgOut[0:w*h])    map(alloc:py1[0:w*h])  map(alloc:py2[0:w*h]) 
  {

#pragma omp target teams distribute parallel for
    for (idx_t i = 0; i < w; i++) {
      real ym1 = 0;
      real ym2 = 0;
      real xm1 = 0;
      for (idx_t j = 0; j < h; j++) {
        py1[i*h+j] = a1 * pimgIn[i*h+j] + a2 * xm1 + b1 * ym1 + b2 * ym2;
        xm1 = pimgIn[i*h+j];
        ym2 = ym1;
        ym1 = py1[i*h+j];
      }
    }

#pragma omp target teams distribute parallel for
    for (idx_t i = 0; i < w; i++) {
      real yp1 = 0;
      real yp2 = 0;
      real xp1 = 0;
      real xp2 = 0;
      for (idx_t j = h - 1; j >= 0; j--) {
        py2[i*h+j] = a3 * xp1 + a4 * xp2 + b1 * yp1 + b2 * yp2;
        xp2 = xp1;
        xp1 = pimgIn[i*h+j];
        yp2 = yp1;
        yp1 = py2[i*h+j];
      }
    }

#pragma omp target teams distribute parallel for  collapse(2)
    for (idx_t i = 0; i < w; i++)
      for (idx_t j = 0; j < h; j++) {
        pimgOut[i*h+j] = c1 * (py1[i*h+j] + py2[i*h+j]);
      }

#pragma omp target teams distribute parallel for
    for (idx_t j = 0; j < h; j++) {
      real tm1 = 0;
      real ym1 = 0;
      real ym2 = 0;
      for (idx_t i = 0; i < w; i++) {
        py1[i*h+j] = a5 * pimgOut[i*h+j] + a6 * tm1 + b1 * ym1 + b2 * ym2;
        tm1 = pimgOut[i*h+j];
        ym2 = ym1;
        ym1 = py1[i*h+j];
      }
    }


#pragma omp target teams distribute parallel for
    for (idx_t j = 0; j < h; j++) {
      real tp1 = 0;
      real tp2 = 0;
      real yp1 = 0;
      real yp2 = 0;
      for (idx_t i = w - 1; i >= 0; i--) {
        py2[i*h+j] = a7 * tp1 + a8 * tp2 + b1 * yp1 + b2 * yp2;
        tp2 = tp1;
        tp1 = pimgOut[i*h+j];
        yp2 = yp1;
        yp1 = py2[i*h+j];
      }
    }

#pragma omp target teams distribute parallel for collapse(2) 
    for (idx_t i = 0; i < w; i++)
      for (idx_t j = 0; j < h; j++)
        pimgOut[i*h+j] = c2 * (py1[i*h+j] + py2[i*h+j]);
  }
}


void run(State &state, pbsize_t pbsize) {
  pbsize_t w = pbsize;                        // 4096
  pbsize_t h = pbsize / 2 + 7 * pbsize / 256; // 2160

  real alpha = 0.25;

  auto imgIn = state.allocate_array<real>({w, h}, /*fakedata*/ true, /*verify*/ false, "imgIn");
  auto imgOut = state.allocate_array<real>({w, h}, /*fakedata*/ false, /*verify*/ true, "imgOut");
  auto y1 = state.allocate_array<real>({w, h}, /*fakedata*/ false, /*verify*/ false, "y1");
  auto y2 = state.allocate_array<real>({w, h}, /*fakedata*/ false, /*verify*/ false, "y2");

  for (auto &&_ : state)
    kernel(w, h, alpha, imgIn, imgOut, y1, y2);
}
