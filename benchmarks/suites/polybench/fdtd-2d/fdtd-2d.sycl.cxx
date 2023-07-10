// BUILD: add_benchmark(ppm=sycl)
#include <CL/sycl.hpp>
#include <rosetta.h>

using namespace cl::sycl;

void mykernel(pbsize_t tmax, pbsize_t nx, pbsize_t ny, buffer<real, 1> &ex, buffer<real, 1> &ey, buffer<real, 1> &hz, buffer<real, 1> &fict, queue &q) {
  for (idx_t t = 0; t < tmax; t++) {
    q.submit([&](handler &cgh) {
      auto ey_acc = ey.get_access<access::mode::read_write>(cgh);
      auto fict_acc = fict.get_access<access::mode::read>(cgh);
      cgh.parallel_for<class ey_update>(range<1>(ny), [=](id<1> idx) {
        ey_acc[idx] = fict_acc[t];
      });
    });

    q.submit([&](handler &cgh) {
      auto ey_acc = ey.get_access<access::mode::read_write>(cgh);
      auto hz_acc = hz.get_access<access::mode::read>(cgh);
      cgh.parallel_for<class ey_hz_update>(range<2>(nx - 1, ny), [=](id<2> idx) {
        idx_t i = idx[0];
        idx_t j = idx[1];
        ey_acc[(i + 1) * ny + j] -= (hz_acc[(i + 1) * ny + j] - hz_acc[i * ny + j]) / 2;
      });
    });

    q.submit([&](handler &cgh) {
      auto ex_acc = ex.get_access<access::mode::read_write>(cgh);
      auto hz_acc = hz.get_access<access::mode::read>(cgh);
      cgh.parallel_for<class ex_hz_update>(range<2>(nx, ny - 1), [=](id<2> idx) {
        idx_t i = idx[0];
        idx_t j = idx[1];
        ex_acc[i * ny + j + 1] -= (hz_acc[i * ny + j + 1] - hz_acc[i * ny + j]) / 2;
      });
    });

    q.submit([&](handler &cgh) {
      auto ex_acc = ex.get_access<access::mode::read>(cgh);
      auto ey_acc = ey.get_access<access::mode::read>(cgh);
      auto hz_acc = hz.get_access<access::mode::read_write>(cgh);
      cgh.parallel_for<class hz_update>(range<2>(nx - 1, ny - 1), [=](id<2> idx) {
        idx_t i = idx[0];
        idx_t j = idx[1];
        hz_acc[i * ny + j] -= 0.7 * (ex_acc[i * ny + j + 1] - ex_acc[i * ny + j] + ey_acc[(i + 1) * ny + j] - ey_acc[i * ny + j]);
      });
    });
  }
  q.wait_and_throw();
}

void run(State &state, pbsize_t pbsize) {
  pbsize_t tmax = 5 * pbsize / 12;   // 500
  pbsize_t nx = pbsize - pbsize / 6; // 1000
  pbsize_t ny = pbsize;              // 1200



  auto ex = state.allocate_array<real>({nx, ny}, /*fakedata*/ true, /*verify*/ true, "ex");
  auto ey = state.allocate_array<real>({nx, ny}, /*fakedata*/ true, /*verify*/ true, "ey");
  auto hz = state.allocate_array<real>({nx, ny}, /*fakedata*/ true, /*verify*/ true, "hz");
  auto fict = state.allocate_array<real>({tmax}, /*fakedata*/ true, /*verify*/ false, "fict");

  queue q(default_selector{});
  {
    buffer<real> ex_buf(ex.data(), range<1>(nx * ny));
    buffer<real> ey_buf(ey.data(), range<1>(nx * ny));
    buffer<real> hz_buf(hz.data(), range<1>(nx * ny));
    buffer<real> fict_buf(fict.data(), range<1>(tmax));
    for (auto &&_ : state) {
      mykernel(tmax, nx, ny, ex_buf, ey_buf, hz_buf, fict_buf, q);
    }
  }
}