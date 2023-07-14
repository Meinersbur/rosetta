// BUILD: add_benchmark(ppm=sycl)
#include <CL/sycl.hpp>
#include <rosetta.h>

using namespace cl::sycl;

static void mykernel(queue &q, pbsize_t tsteps, pbsize_t n, buffer<real> &A_buf) {

  for (pbsize_t t = 0; t <= tsteps - 1; t++) {
    q.submit([&](handler &cgh) {
      auto A_acc = A_buf.get_access<access::mode::read_write>(cgh);

      cgh.parallel_for<class kernelFunc>(range<2>(n - 1, n - 1), [=](id<2> idx) {
        idx_t i = idx[0] + 1;
        idx_t j = idx[1] + 1;
        A_acc[i * n + j] = (A_acc[(i - 1) * n + j - 1] +
                            A_acc[(i - 1) * n + j] +
                            A_acc[(i - 1) * n + j + 1] +
                            A_acc[i * n + j - 1] +
                            A_acc[i * n + j] +
                            A_acc[i * n + j + 1] +
                            A_acc[(i + 1) * n + j - 1] +
                            A_acc[(i + 1) * n + j] +
                            A_acc[(i + 1) * n + j + 1]) /
                           9;
      });
    });
  }
  q.wait_and_throw();
}

void run(State &state, int pbsize) {
  pbsize_t tsteps = 1; // 500
  pbsize_t n = pbsize; // 2000


  //Changed verify to false. Result is non-deterministic by the algorithm by design
  auto A = state.allocate_array<real>({n, n}, /*fakedata*/ true, /*verify*/ false, "A");
  queue q(default_selector{});
  {
    buffer<real, 1> A_buf(A.data(), range<1>(n * n));
    for (auto &&_ : state) {
      mykernel(q, tsteps, n, A_buf);
    }
  }
}
