// BUILD: add_benchmark(ppm=hip)

#include "rosetta.h"

static unsigned num_blocks(int num, int factor) {
  return (num + factor - 1) / factor;
}


__global__ void kernel_y(pbsize_t n,
                         real alpha, real beta,
                         real *A,
                         real *B,
                         real tmp[],
                         real x[],
                         real y[]) {
  idx_t i = blockDim.x * blockIdx.x + threadIdx.x;



  if (i < n) {
    tmp[i] = 0;
    y[i] = 0;
    for (idx_t j = 0; j < n; j++) {
      tmp[i] += A[i * n + j] * x[j];
      y[i] += B[i * n + j] * x[j];
    }
    y[i] = alpha * tmp[i] + beta * y[i];
  }
}



static void kernel(pbsize_t n,
                   real alpha, real beta,
                   real *A,
                   real *B,
                   real tmp[],
                   real x[],
                   real y[]) {
  const unsigned threadsPerBlock = 256;
  kernel_y<<<threadsPerBlock, num_blocks(n, threadsPerBlock)>>>(n, alpha, beta, A, B, tmp, x, y);
}



void run(State &state, pbsize_t n) {
  real alpha = 1.5;
  real beta = 1.2;
  auto A = state.allocate_array<double>({n, n}, /*fakedata*/ true, /*verify*/ false, "A");
  auto B = state.allocate_array<double>({n, n}, /*fakedata*/ true, /*verify*/ false, "B");
  auto x = state.allocate_array<double>({n}, /*fakedata*/ true, /*verify*/ false, "x");
  auto y = state.allocate_array<double>({n}, /*fakedata*/ false, /*verify*/ true, "y");


  real *dev_A = state.allocate_dev_hip<real>(n * n);
  real *dev_B = state.allocate_dev_hip<real>(n * n);
  real *dev_tmp = state.allocate_dev_hip<real>(n);
  real *dev_x = state.allocate_dev_hip<real>(n);
  real *dev_y = state.allocate_dev_hip<real>(n);

  for (auto &&_ : state) {
    BENCH_HIP_TRY(hipMemcpy(dev_A, A.data(), n * n * sizeof(real), hipMemcpyHostToDevice));
    BENCH_HIP_TRY(hipMemcpy(dev_B, B.data(), n * n * sizeof(real), hipMemcpyHostToDevice));
    BENCH_HIP_TRY(hipMemcpy(dev_x, x.data(), n * sizeof(real), hipMemcpyHostToDevice));

    kernel(n, alpha, beta, dev_A, dev_B, dev_tmp, dev_x, dev_y);

    BENCH_HIP_TRY(hipMemcpy(y.data(), dev_y, n * sizeof(real), hipMemcpyDeviceToHost));

    BENCH_HIP_TRY(hipDeviceSynchronize());
  }

  state.free_dev_hip(dev_A);
  state.free_dev_hip(dev_B);
  state.free_dev_hip(dev_tmp);
  state.free_dev_hip(dev_x);
  state.free_dev_hip(dev_y);
}
