// BUILD: add_benchmark(ppm=sycl)
#include <CL/sycl.hpp>
#include <rosetta.h>

using namespace cl::sycl;

static void mykernel(queue &q, pbsize_t w, pbsize_t h, real alpha,
                     buffer<real, 1> &imgIn_buf, buffer<real, 1> &imgOut_buf,
                     buffer<real, 1> &y1_buf, buffer<real, 1> &y2_buf) {

  real k = (1 - std::exp(-alpha)) * (1 - std::exp(-alpha)) / (1 + 2 * alpha * std::exp(-alpha) - std::exp(2 * alpha));
  real a1 = k, a5 = k;
  real a6 = k * std::exp(-alpha) * (alpha - 1), a2 = a6;
  real a7 = k * std::exp(-alpha) * (alpha + 1), a3 = a7;
  real a8 = -k * std::exp(-2 * alpha), a4 = a8;
  real b1 = std::pow(2, -alpha);
  real b2 = -std::exp(-2 * alpha);
  real c1 = 1, c2 = 1;

  q.submit([&](handler &cgh) {
    auto imgIn_acc = imgIn_buf.get_access<access::mode::read>(cgh);
    auto y1_acc = y1_buf.get_access<access::mode::read_write>(cgh);
    cgh.parallel_for<class compute_y1>(range<1>(w), [=](id<1> i) {
      real ym1 = 0;
      real ym2 = 0;
      real xm1 = 0;
      for (idx_t j = 0; j < h; j++) {
        y1_acc[i * h + j] = a1 * imgIn_acc[i * h + j] + a2 * xm1 + b1 * ym1 + b2 * ym2;
        xm1 = imgIn_acc[i * h + j];
        ym2 = ym1;
        ym1 = y1_acc[i * h + j];
      }
    });
  });
  q.submit([&](handler &cgh) {
    auto imgIn_acc = imgIn_buf.get_access<access::mode::read>(cgh);
    auto y2_acc = y2_buf.get_access<access::mode::read_write>(cgh);
    cgh.parallel_for<class compute_y2>(range<1>(w), [=](id<1> i) {
      real yp1 = 0;
      real yp2 = 0;
      real xp1 = 0;
      real xp2 = 0;
      for (idx_t j = h - 1; j >= 0; j--) {
        y2_acc[i * h + j] = a3 * xp1 + a4 * xp2 + b1 * yp1 + b2 * yp2;
        xp2 = xp1;
        xp1 = imgIn_acc[i * h + j];
        yp2 = yp1;
        yp1 = y2_acc[i * h + j];
      }
    });
  });
  q.submit([&](handler &cgh) {
    auto y1_acc = y1_buf.get_access<access::mode::read>(cgh);
    auto y2_acc = y2_buf.get_access<access::mode::read>(cgh);
    auto imgOut_acc = imgOut_buf.get_access<access::mode::write>(cgh);
    cgh.parallel_for<class compute_imgOut_1>(range<2>(w, h), [=](id<2> idx) {
      idx_t i = idx[0];
      idx_t j = idx[1];
      imgOut_acc[i * h + j] = c1 * (y1_acc[i * h + j] + y2_acc[i * h + j]);
    });
  });
  q.submit([&](handler &cgh) {
    auto imgOut_acc = imgOut_buf.get_access<access::mode::read>(cgh);
    auto y1_acc = y1_buf.get_access<access::mode::read_write>(cgh);
    cgh.parallel_for<class compute_y1_2>(range<1>(h), [=](id<1> j) {
      real tm1 = 0;
      real ym1 = 0;
      real ym2 = 0;
      for (idx_t i = 0; i < w; i++) {
        y1_acc[i * h + j] = a5 * imgOut_acc[i * h + j] + a6 * tm1 + b1 * ym1 + b2 * ym2;
        tm1 = imgOut_acc[i * h + j];
        ym2 = ym1;
        ym1 = y1_acc[i * h + j];
      }
    });
  });
  q.submit([&](handler &cgh) {
    auto imgOut_acc = imgOut_buf.get_access<access::mode::read>(cgh);
    auto y2_acc = y2_buf.get_access<access::mode::read_write>(cgh);
    cgh.parallel_for<class compute_y2_2>(range<1>(h), [=](id<1> j) {
      real tp1 = 0;
      real tp2 = 0;
      real yp1 = 0;
      real yp2 = 0;
      for (idx_t i = w - 1; i >= 0; i--) {
        y2_acc[i * h + j] = a7 * tp1 + a8 * tp2 + b1 * yp1 + b2 * yp2;
        tp2 = tp1;
        tp1 = imgOut_acc[i * h + j];
        yp2 = yp1;
        yp1 = y2_acc[i * h + j];
      }
    });
  });
  q.submit([&](handler &cgh) {
    auto y1_acc = y1_buf.get_access<access::mode::read>(cgh);
    auto y2_acc = y2_buf.get_access<access::mode::read>(cgh);
    auto imgOut_acc = imgOut_buf.get_access<access::mode::write>(cgh);
    cgh.parallel_for<class compute_imgOut_2>(range<2>(w, h), [=](id<2> idx) {
      idx_t i = idx[0];
      idx_t j = idx[1];
      imgOut_acc[i * h + j] = c2 * (y1_acc[i * h + j] + y2_acc[i * h + j]);
    });
  });
  q.wait_and_throw();
}

void run(State &state, pbsize_t pbsize) {
  pbsize_t w = pbsize;                        // 4096
  pbsize_t h = pbsize / 2 + 7 * pbsize / 256; // 2160

  real alpha = 0.25;

  auto imgIn = state.allocate_array<real>({w, h}, /*fakedata*/ true, /*verify*/ false, "imgIn");
  auto imgOut = state.allocate_array<real>({w, h}, /*fakedata*/ false, /*verify*/ true, "imgOut");

  queue q(default_selector{});
  {
    buffer<real, 1> imgIn_buf(imgIn.data(), range<1>(w * h));
    buffer<real, 1> imgOut_buf(imgOut.data(), range<1>(w * h));
    buffer<real, 1> y1_buf(w * h);
    buffer<real, 1> y2_buf(w * h);
    for (auto &&_ : state) {
      mykernel(q, w, h, alpha, imgIn_buf, imgOut_buf, y1_buf, y2_buf);
    }
  }
}