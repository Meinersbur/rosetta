// BUILD: add_benchmark(ppm=omp_parallel)

#include "rosetta.h"



static void kernel(pbsize_t w, pbsize_t h,
                   real alpha,
                   multarray<real, 2> imgIn,
                   multarray<real, 2> imgOut,
                   multarray<real, 2> y1,
                   multarray<real, 2> y2) {
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

#pragma omp parallel default(none) firstprivate(w,h,imgIn,imgOut,y1,y2) firstprivate(k,a1,a5,a6,a2,a7,a3,a8,a4,b1,b2,c1,c2)
                       {
                   
#pragma omp for schedule(static)
                           for (idx_t i = 0; i < w; i++) {
                               real ym1 = 0;
                               real ym2 = 0;
                               real xm1 = 0;
                               for (idx_t j = 0; j < h; j++) {
                                   y1[i][j] = a1 * imgIn[i][j] + a2 * xm1 + b1 * ym1 + b2 * ym2;
                                   xm1 = imgIn[i][j];
                                   ym2 = ym1;
                                   ym1 = y1[i][j];
                               }
                           }

#pragma omp for schedule(static)
                           for (idx_t i = 0; i < w; i++) {
                               real yp1 = 0;
                               real yp2 = 0;
                               real xp1 = 0;
                               real xp2 = 0;
                               for (idx_t j = h - 1; j >= 0; j--) {
                                   y2[i][j] = a3 * xp1 + a4 * xp2 + b1 * yp1 + b2 * yp2;
                                   xp2 = xp1;
                                   xp1 = imgIn[i][j];
                                   yp2 = yp1;
                                   yp1 = y2[i][j];
                               }
                           }

#pragma omp for collapse(2) schedule(static)
                           for (idx_t i = 0; i < w; i++)
                               for (idx_t j = 0; j < h; j++) {
                                   imgOut[i][j] = c1 * (y1[i][j] + y2[i][j]);
                               }

#pragma omp for schedule(static)
                           for (idx_t j = 0; j < h; j++) {
                               real tm1 = 0;
                               real ym1 = 0;
                               real ym2 = 0;
                               for (idx_t i = 0; i < w; i++) {
                                   y1[i][j] = a5 * imgOut[i][j] + a6 * tm1 + b1 * ym1 + b2 * ym2;
                                   tm1 = imgOut[i][j];
                                   ym2 = ym1;
                                   ym1 = y1[i][j];
                               }
                           }


#pragma omp for schedule(static)
                           for (idx_t j = 0; j < h; j++) {
                               real tp1 = 0;
                               real tp2 = 0;
                               real yp1 = 0;
                               real yp2 = 0;
                               for (idx_t i = w - 1; i >= 0; i--) {
                                   y2[i][j] = a7 * tp1 + a8 * tp2 + b1 * yp1 + b2 * yp2;
                                   tp2 = tp1;
                                   tp1 = imgOut[i][j];
                                   yp2 = yp1;
                                   yp1 = y2[i][j];
                               }
                           }

#pragma omp for collapse(2) schedule(static)
                           for (idx_t i = 0; i < w; i++)
                               for (idx_t j = 0; j < h; j++)
                                   imgOut[i][j] = c2 * (y1[i][j] + y2[i][j]);
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
