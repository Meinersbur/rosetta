// BUILD: add_benchmark(ppm=cuda)

#include <rosetta.h>

#include <thrust/reduce.h>
#include <thrust/transform_reduce.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/functional.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/for_each.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>



struct Lij_times_xj : public thrust::unary_function<real, real> {
  pbsize_t n;
  thrust::device_ptr<real> L;
  thrust::device_ptr<real> x;
  idx_t i;

  Lij_times_xj(pbsize_t n, thrust::device_ptr<real> L, thrust::device_ptr<real> x, idx_t i) : n(n), L(L), x(x), i(i) {}

  __host__ __device__ real operator()(pbsize_t j) const {
     real v = L[i * n + j] * x[j];    
    //  real v = L[i * n + j];   

      return v;
  }
};


static void kernel(pbsize_t n, thrust::device_ptr<real> L, thrust::device_ptr<real> x, real b[], real *host_L) {
#if 0
    for (idx_t i = 0; i < n; i++) {
        x[i] = b[i];
        for (idx_t j = 0; j < i; j++)
            x[i] -= L[i][j] * x[j];
        x[i] /= L[i][i];
    }
#endif  
    fprintf(stderr, "Alive!\n");

  for (idx_t i = 0; i < n; i++) {
    Lij_times_xj op{n, L, x, i};
    real sum = thrust::transform_reduce(
        thrust::device,
        thrust::make_counting_iterator(0),
        thrust::make_counting_iterator(0) + i,
        op,
        (real)0,
        thrust::plus<real>());
    x[i] = (b[i] - sum) / L[i * n + i] ;
   // fprintf(stderr, "sum = %f\n", sum);
   // assert(L[0] == host_L[0]);
   // x[i] = 1 +sum;
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
  auto x = state.allocate_array<real>({n}, /*fakedata*/ false, /*verify*/ true, "x");
  auto b = state.allocate_array<real>({n}, /*fakedata*/ true, /*verify*/ false, "b");

  real *dev_L = state.allocate_dev<real>(n * n);
  real *dev_x = state.allocate_dev<real>(n);
  // real* dev_b = state.allocate_dev<real>(n);

  for (auto &&_ : state) {
    BENCH_CUDA_TRY(cudaMemcpy(dev_L, L.data(), n * n * sizeof(real), cudaMemcpyHostToDevice));
    BENCH_CUDA_TRY(cudaMemcpy(dev_x, x.data(), n * sizeof(real), cudaMemcpyHostToDevice));
    // BENCH_CUDA_TRY( cudaMemcpy(dev_b, b.data(), n* sizeof(real), cudaMemcpyHostToDevice));
    kernel(n, thrust::device_pointer_cast(dev_L), thrust::device_pointer_cast(dev_x), b, &L.get()[0][0]);
    BENCH_CUDA_TRY(cudaMemcpy(x.data(), dev_x, n * sizeof(real), cudaMemcpyDeviceToHost));

    BENCH_CUDA_TRY(cudaDeviceSynchronize());
  }

  state.free_dev(dev_L);
  state.free_dev(dev_x);
  // state.free_dev(dev_b);
}
