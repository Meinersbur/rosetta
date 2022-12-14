// BUILD: add_benchmark(ppm=cuda)

#include <rosetta.h>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>



struct Lij_times_xj : public thrust::unary_function<real, real> {
  pbsize_t n;
  thrust::device_ptr<real> L;
  thrust::device_ptr<real> x;
  idx_t i;

  Lij_times_xj(pbsize_t n, thrust::device_ptr<real> L, thrust::device_ptr<real> x, idx_t i) : n(n), L(L), x(x), i(i) {}

  __host__ __device__ real operator()(pbsize_t j) const {
    return L[i * n + j] * x[j];
  }
};


static void kernel(pbsize_t n,
                   thrust::device_ptr<real> L, thrust::device_ptr<real> x, real b[]) {


  for (idx_t i = 0; i < n; i++) {
    Lij_times_xj op{n, L, x, i};
    real sum = thrust::transform_reduce(
        thrust::device,
        thrust::make_counting_iterator(0),
        thrust::make_counting_iterator(0) + i,
        op,
        0,
        thrust::plus<real>());
    x[i] = (b[i] - sum) / L[i * n + i];
  }

#if 0
  for (idx_t i = 0; i < n; i++) {
    real sum = 0 ;
#pragma omp parallel for schedule(static) default(none) firstprivate(i, x, L) reduction(+ \
                                                                                        : sum)
    for (idx_t j = 0; j < i; j++)
      sum += L[i][j] * x[j];
    x[i] =  (b[i] - sum) / L[i][i];
  }
#endif
}


void run(State &state, pbsize_t pbsize) {
  pbsize_t n = pbsize; // 2000


  auto L = state.allocate_array<real>({n, n}, /*fakedata*/ true, /*verify*/ false, "L");
  auto x = state.allocate_array<real>({n}, /*fakedata*/ true, /*verify*/ true, "x");
  auto b = state.allocate_array<real>({n}, /*fakedata*/ true, /*verify*/ false, "b");

  real *dev_L = state.allocate_dev<real>(n * n);
  real *dev_x = state.allocate_dev<real>(n);
  // real* dev_b = state.allocate_dev<real>(n);

  for (auto &&_ : state) {
    BENCH_CUDA_TRY(cudaMemcpy(dev_L, L.data(), n * n * sizeof(real), cudaMemcpyHostToDevice));
    BENCH_CUDA_TRY(cudaMemcpy(dev_x, x.data(), n * sizeof(real), cudaMemcpyHostToDevice));
    // BENCH_CUDA_TRY( cudaMemcpy(dev_b, b.data(), n* sizeof(real), cudaMemcpyHostToDevice));
    kernel(n, thrust::device_pointer_cast(dev_L), thrust::device_pointer_cast(dev_x), b);
    BENCH_CUDA_TRY(cudaMemcpy(x.data(), dev_x, n * sizeof(real), cudaMemcpyDeviceToHost));

    BENCH_CUDA_TRY(cudaDeviceSynchronize());
  }

  state.free_dev(dev_L);
  state.free_dev(dev_x);
  // state.free_dev(dev_b);
}
