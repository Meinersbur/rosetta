// BUILD: add_benchmark(ppm=sycl)
#include <CL/sycl.hpp>
#include <rosetta.h>

using namespace cl::sycl;

static void mykernel(queue &q, pbsize_t m, pbsize_t n, buffer<real> &data_buf, buffer<real> &cov_buf, buffer<real> &mean_buf) {

  q.submit([&](handler &cgh) {
    auto data_acc = data_buf.get_access<access::mode::read>(cgh);
    auto mean_acc = mean_buf.get_access<access::mode::write>(cgh);

    cgh.parallel_for<class kernel_mean>(range<1>(m), [=](id<1> j) {
      mean_acc[j] = 0.0;
      for (idx_t i = 0; i < n; i++)
        mean_acc[j] += data_acc[i * m + j];
      mean_acc[j] /= n;
    });
  });
  q.submit([&](handler &cgh) {
    auto data_acc = data_buf.get_access<access::mode::read_write>(cgh);
    auto mean_acc = mean_buf.get_access<access::mode::read>(cgh);

    cgh.parallel_for<class kernel_reduce>(range<2>(n, m), [=](id<2> idx) {
      idx_t i = idx[0];
      idx_t j = idx[1];
      data_acc[i * m + j] -= mean_acc[j];
    });
  });
  q.submit([&](handler &cgh) {
    auto data_acc = data_buf.get_access<access::mode::read>(cgh);
    auto cov_acc = cov_buf.get_access<access::mode::write>(cgh);

    cgh.parallel_for<class kernel_covariance>(range<2>(m, m), [=](id<2> idx) {
      idx_t i = idx[0];
      idx_t j = idx[1];
      if (i < m && j < m) {
        cov_acc[i * m + j] = 0.0;
        for (idx_t k = 0; k < n; k++)
          cov_acc[i * m + j] += data_acc[k * m + i] * data_acc[k * m + j];
        cov_acc[i * m + j] /= (n - 1.0);
        cov_acc[j * m + i] = cov_acc[i * m + j];
      }
    });
  });
  q.wait_and_throw();
}

void run(State &state, pbsize_t pbsize) {
  pbsize_t n = pbsize;
  pbsize_t m = pbsize - pbsize / 8;

  auto data = state.allocate_array<real>({n, m}, /*fakedata*/ true, /*verify*/ false, "data");
  auto mean = state.allocate_array<real>({m}, /*fakedata*/ false, /*verify*/ true, "mean");
  auto cov = state.allocate_array<real>({m, m}, /*fakedata*/ false, /*verify*/ true, "cov");

  queue q(default_selector{});
  {
    buffer<real> data_buf(data.data(), range<1>(n * m));
    buffer<real> cov_buf(cov.data(), range<1>(m * m));
    buffer<real> mean_buf(mean.data(), range<1>(m));
    for (auto &&_ : state) {
      mykernel(q, m, n, data_buf, cov_buf, mean_buf);
    }
  }
}