// BUILD: add_benchmark(ppm=hip)

#include <rosetta.h>



__global__ void kernel3(pbsize_t m, pbsize_t n, real *A, real *x, real *y, real *tmp) {
  idx_t i = blockDim.x * blockIdx.x + threadIdx.x;

  if (i < m) {
    for (idx_t j = 0; j < n; j++)
      tmp[i] += A[i * n + j] * x[j];
  }
}


__global__ void kernel4(pbsize_t m, pbsize_t n, real *A, real *x, real *y, real *tmp) {
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


  real *dev_A = state.allocate_dev_hip<real>(n * m);
  real *dev_x = state.allocate_dev_hip<real>(n);
  real *dev_y = state.allocate_dev_hip<real>(n);
  real *dev_tmp = state.allocate_dev_hip<real>(m);



  for (auto &&_ : state) {
    hipMemcpy(dev_A, A.data(), n * m * sizeof(real), hipMemcpyHostToDevice);
    hipMemcpy(dev_x, A.data(), n * sizeof(real), hipMemcpyHostToDevice);
    hipMemset(dev_y, '\0', n * sizeof(real));
    hipMemset(dev_tmp, '\0', m * sizeof(real));


    const int threadsPerBlock = 256;
    kernel3<<<threadsPerBlock, num_blocks(m, threadsPerBlock)>>>(m, n, dev_A, dev_x, dev_y, dev_tmp);
    kernel4<<<threadsPerBlock, num_blocks(n, threadsPerBlock)>>>(m, n, dev_A, dev_x, dev_y, dev_tmp);

    hipMemcpy(y.data(), dev_y, n * sizeof(real), hipMemcpyDeviceToHost);

    hipDeviceSynchronize();
  }


  state.free_dev_hip(dev_A);
  state.free_dev_hip(dev_x);
  state.free_dev_hip(dev_y);
  state.free_dev_hip(dev_tmp);
}
