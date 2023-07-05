// BUILD: add_benchmark(ppm=sycl)
#include <CL/sycl.hpp>
#include <rosetta.h>

using namespace cl::sycl;

void mykernel(queue &q, pbsize_t m, pbsize_t n, buffer<real> &data_buf, buffer<real> &corr_buf, buffer<real> &mean_buf, buffer<real> &stddev_buf) {

    q.submit([&](handler &cgh) {

        auto data_acc = data_buf.get_access<access::mode::read>(cgh);
        auto mean_acc = mean_buf.get_access<access::mode::read_write>(cgh);
        auto stddev_acc = stddev_buf.get_access<access::mode::read_write>(cgh);
        real eps = 0.1;

        cgh.parallel_for(range<1>(m), [=](id<1> j) {
            mean_acc[j] = 0.0;
            for (idx_t i = 0; i < n; i++)
                mean_acc[j] += data_acc[i*m +j];
            mean_acc[j] /= n;
            stddev_acc[j] = 0.0;
            for (idx_t i = 0; i < n; i++)
                stddev_acc[j] += (data_acc[i*m +j] - mean_acc[j]) * (data_acc[i*m +j] - mean_acc[j]);
            stddev_acc[j] /= n;
            stddev_acc[j] = sqrt(stddev_acc[j]);
            if (stddev_acc[j] <= eps)
                stddev_acc[j] = 1.0;
        });
    });

    q.submit([&](handler &cgh) {

        auto data_acc = data_buf.get_access<access::mode::write>(cgh);
        auto mean_acc = mean_buf.get_access<access::mode::read>(cgh);
        auto stddev_acc = stddev_buf.get_access<access::mode::read>(cgh);

        cgh.parallel_for(range<2>(n,m), [=](id<2> idx) {
            idx_t i = idx[0];
            idx_t j = idx[1];
            data_acc[i*m +j] -= mean_acc[j];
            data_acc[i*m +j] /= sqrt((real)n) * stddev_acc[j];
        });
    });

    q.submit([&](handler &cgh) {

        auto data_acc = data_buf.get_access<access::mode::read>(cgh);
        auto corr_acc = corr_buf.get_access<access::mode::read_write>(cgh);

        cgh.parallel_for(range<2>(m,m), [=](id<2> idx) {
            idx_t i = idx[0];
            idx_t j = idx[1];
            if(i == j){
                corr_acc[i*m +j] = 1.0;
            }
            if(i<j){
                for (idx_t k = 0; k < n; k++)
                    corr_acc[i*m +j] += (data_acc[k*m + i] * data_acc[k*m + j]);
                corr_acc[j*m + i] = corr_acc[i*m +j];
            }
        });
    });
    q.wait_and_throw();
}

void run(State &state, pbsize_t pbsize) {
  pbsize_t n = pbsize;
  pbsize_t m = pbsize - pbsize / 6;


  auto data = state.allocate_array<real>({n, m}, /*fakedata*/ true, /*verify*/ false, "data");
  auto mean = state.allocate_array<real>({m}, /*fakedata*/ false, /*verify*/ true, "mean");
  auto stddev = state.allocate_array<real>({m}, /*fakedata*/ false, /*verify*/ true, "stddev");
  auto corr = state.allocate_array<real>({m, m}, /*fakedata*/ false, /*verify*/ true, "corr");


queue q(default_selector{});
  {
    buffer<real, 1> data_buf(data.data(),range<1>(n* m));
    buffer<real, 1> corr_buf(corr.data(),range<1>(m* m));
    buffer<real, 1> mean_buf(mean.data(),range<1>(m));
    buffer<real, 1> stddev_buf(stddev.data(),range<1>(m));
    
    for (auto &&_ : state) {
      mykernel(q, m, n, data_buf, corr_buf, mean_buf,stddev_buf);
    }
  }
}