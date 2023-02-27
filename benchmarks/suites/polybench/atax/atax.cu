// BUILD: add_benchmark(ppm=cuda)
#include "rosetta.h"



__global__ void kernel3(int m, int n, real *A, real *x, real *y, real *tmp) {
  idx_t i = blockDim.x * blockIdx.x + threadIdx.x;

  if (i < m) {
    for (idx_t j = 0; j < n; j++)
      tmp[i] += A[i * n + j] * x[j];
  }
}


__global__ void kernel4(int m, int n, real *A, real *x, real *y, real *tmp) {
  idx_t j = blockDim.x * blockIdx.x + threadIdx.x;

  if (j < n) {
    for (idx_t i = 0; i < m; i++)
      y[j] += A[i * n + j] * tmp[i];
  }
}



static int num_blocks(int num, int factor) {
  return (num + factor - 1) / factor;
}


void run(State &state, int pbsize) {
  // n is 5%-20% larger than m
  pbsize_t n = pbsize;
  pbsize_t m = pbsize - pbsize / 10;


  auto A = state.allocate_array<real>({m, n}, /*fakedata*/ true, /*verify*/ false, "A");
  auto x = state.allocate_array<real>({n}, /*fakedata*/ true, /*verify*/ false, "x");
  auto y = state.allocate_array<real>({n}, /*fakedata*/ false, /*verify*/ true, "y");


  real *dev_A = state.allocate_dev<real>(n * m);
  real *dev_x = state.allocate_dev<real>(n);
  real *dev_y = state.allocate_dev<real>(n);
  real *dev_tmp = state.allocate_dev<real>(m);



  for (auto &&_ : state) {
    cudaMemcpy(dev_A, A.data(), n * m * sizeof(real), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_x, A.data(), n * sizeof(real), cudaMemcpyHostToDevice);
    cudaMemset(dev_y, '\0', n * sizeof(real));
    cudaMemset(dev_tmp, '\0', m * sizeof(real));


    const int threadsPerBlock = 256;
    kernel3<<<threadsPerBlock, num_blocks(m, threadsPerBlock)>>>(m, n, dev_A, dev_x, dev_y, dev_tmp);
    kernel4<<<threadsPerBlock, num_blocks(n, threadsPerBlock)>>>(m, n, dev_A, dev_x, dev_y, dev_tmp);

    cudaMemcpy(y.data(), dev_y, n * sizeof(real), cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();
  }


  state.free_dev(dev_A);
  state.free_dev(dev_x);
  state.free_dev(dev_y);
  state.free_dev(dev_tmp);
}
