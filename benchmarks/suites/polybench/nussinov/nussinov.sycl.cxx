// BUILD: add_benchmark(ppm=sycl)
#include <CL/sycl.hpp>
#include <rosetta.h>

using namespace cl::sycl;

void myKernel(pbsize_t n, buffer<real, 1> &seq_buf, buffer<real, 1> &table_buf, queue &q) {
  for (idx_t w = n; w < 2 * n - 1; ++w) {
    q.submit([&](handler &cgh) {
      auto seq = seq_buf.get_access<access::mode::read>(cgh);
      auto table = table_buf.get_access<access::mode::read_write>(cgh);

      cgh.parallel_for<class kernel_max_score>(range<1>(n), [=](id<1> idx) {
        auto j = idx[0];
        auto i = ((idx_t)n - 1) + j - w;

        if (0 <= i && i < n && i + 1 <= j && j < n) {
          real maximum = table[i * n + j];

          if (j - 1 >= 0)
            maximum = sycl::max(maximum, table[i * n + j - 1]);
          if (i + 1 < n)
            maximum = sycl::max(maximum, table[(i + 1) * n + j]);

          if (j - 1 >= 0 && i + 1 < n) {
            auto upd = table[(i + 1) * n + j - 1];
            if (i < j - 1)
              upd += (seq[i] + seq[j] == 3) ? (real)1 : (real)0;

            maximum = sycl::max(maximum, upd);
          }

          for (idx_t k = i + 1; k < j; k++)
            maximum = sycl::max(maximum, table[i * n + k] + table[(k + 1) * n + j]);
          table[i * n + j] = maximum;
        }
      });
    });
  }
  q.wait_and_throw();
}

void run(State &state, pbsize_t pbsize) {
  pbsize_t n = pbsize; // 2500



  auto seq = state.allocate_array<real>({n}, /*fakedata*/ true, /*verify*/ false, "seq");
  auto table = state.allocate_array<real>({n, n}, /*fakedata*/ true, /*verify*/ true, "table");

  queue q(default_selector{});
  {
    buffer<real, 1> seq_buf(seq.data(), range<1>(n));
    buffer<real, 1> table_buf(table.data(), range<1>(n * n));
    for (auto &&_ : state) {
      myKernel(n, seq_buf, table_buf, q);
    }
  }
}