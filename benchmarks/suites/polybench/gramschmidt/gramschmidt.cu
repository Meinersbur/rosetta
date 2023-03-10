// BUILD: add_benchmark(ppm=cuda,sources=[__file__, "gramschmidt-common.cxx"])

#include "gramschmidt-common.h"
#include <rosetta.h>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>



static unsigned num_blocks(int num, int factor) {
  return (num + factor - 1) / factor;
}


__device__ real sqr(real v) { return v * v; }



template <typename T>
struct outer_sqr : public thrust::unary_function<T, T> {
  pbsize_t m;
  pbsize_t n;
  T *A;
  idx_t k;

  outer_sqr(pbsize_t m, pbsize_t n, T *A, idx_t k) : m(m), n(n), A(A), k(k) {}


  __device__ T operator()(int i) const {
    return sqr(A[i * n + k]);
  }
};



__global__ void kernel_Q(pbsize_t m, pbsize_t n,
                         real *A, real *R, real *Q, idx_t k) {
  idx_t i = blockDim.x * blockIdx.x + threadIdx.x;

  if (i < m)
    Q[i * n + k] = A[i * n + k] / R[k * n + k];
}



__global__ void kernel_R(pbsize_t m, pbsize_t n,
                         real *A, real *R, real *Q, idx_t k) {
  idx_t j = blockDim.x * blockIdx.x + threadIdx.x + k + 1;

  if (j < n) {
    R[k * n + j] = 0;
    for (idx_t i = 0; i < m; i++)
      R[k * n + j] += Q[i * n + k] * A[i * n + j];
  }
}


__global__ void kernel_A(pbsize_t m, pbsize_t n,
                         real *A, real *R, real *Q, idx_t k) {
  idx_t j = blockDim.x * blockIdx.x + threadIdx.x + k + 1;

  if (j < n) {
    for (idx_t i = 0; i < m; i++)
      A[i * n + j] -= Q[i * n + k] * R[k * n + j];
  }
}



static void kernel(pbsize_t m, pbsize_t n,
                   real *A, real *R, real *Q) {
  const unsigned threadsPerBlock = 256;
  thrust::device_ptr<real> wrapped_A = thrust::device_pointer_cast(A);
  thrust::device_ptr<real> wrapped_R = thrust::device_pointer_cast(R);
  thrust::device_ptr<real> wrapped_Q = thrust::device_pointer_cast(Q);

  for (idx_t k = 0; k < n; k++) {
    outer_sqr<real> op{m, n, A, k};
    real sum =
        thrust::transform_reduce(thrust::device,
                                 thrust::make_counting_iterator(0),
                                 thrust::make_counting_iterator(0) + m,
                                 op,
                                 (real)0,
                                 thrust::plus<real>());




    wrapped_R[k * n + k] = std::sqrt(sum);


    kernel_Q<<<threadsPerBlock, num_blocks(m, threadsPerBlock)>>>(m, n, A, R, Q, k);
    kernel_R<<<threadsPerBlock, num_blocks(n - (k + 1), threadsPerBlock)>>>(m, n, A, R, Q, k);
    kernel_A<<<threadsPerBlock, num_blocks(n - (k + 1), threadsPerBlock)>>>(m, n, A, R, Q, k);
  }

}



void run(State &state, pbsize_t pbsize) {
    pbsize_t m = pbsize;              // 1200
    pbsize_t n = pbsize - pbsize / 6; // 1000


  auto A = state.allocate_array<real>({m, n}, /*fakedata*/ true, /*verify*/ true, "A");
  auto R = state.allocate_array<real>({n, n}, /*fakedata*/ false, /*verify*/ true, "R");
  auto Q = state.allocate_array<real>({m, n}, /*fakedata*/ false, /*verify*/ true, "Q");

  real *dev_A = state.allocate_dev<real>(m * n);
  real *dev_R = state.allocate_dev<real>(n * n);
  real *dev_Q = state.allocate_dev<real>(m * n);

  for (auto &&_ : state.manual()) {
    condition(m, n, A);
    {
      auto &&scope = _.scope();

      BENCH_CUDA_TRY(cudaMemcpy(dev_A, A.data(), m * n * sizeof(real), cudaMemcpyHostToDevice));


      kernel(m, n, dev_A, dev_R, dev_Q);

      BENCH_CUDA_TRY(cudaMemcpy(A.data(), dev_A, m * n * sizeof(real), cudaMemcpyDeviceToHost));
      BENCH_CUDA_TRY(cudaMemcpy(R.data(), dev_R, n * n * sizeof(real), cudaMemcpyDeviceToHost));
      BENCH_CUDA_TRY(cudaMemcpy(Q.data(), dev_Q, m * n * sizeof(real), cudaMemcpyDeviceToHost));

      BENCH_CUDA_TRY(cudaDeviceSynchronize());
    }
  }

  state.free_dev(dev_A);
  state.free_dev(dev_R);
  state.free_dev(dev_Q);
}
