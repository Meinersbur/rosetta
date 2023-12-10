// BUILD: add_benchmark(ppm=hip)

#include "rosetta.h"



__global__ void kernel_dev(pbsize_t ni, pbsize_t nj, pbsize_t nk,
                           real alpha,
                           real beta,
                           real *C, real *A, real *B) {
  idx_t i = blockDim.x * blockIdx.x + threadIdx.x;
  idx_t j = blockDim.y * blockIdx.y + threadIdx.y;


  if (i < ni && j < nj) {
    C[i * nj + j] *= beta;


    for (idx_t k = 0; k < nk; k++)
      C[i * nj + j] += alpha * A[i * nk + k] * B[k * nj + j];
  }
}



static unsigned num_blocks(int num, int factor) {
  return (num + factor - 1) / factor;
}



static void kernel(pbsize_t ni, pbsize_t nj, pbsize_t nk,
                   real alpha,
                   real beta,
                   real *C, real *A, real *B) {

  unsigned threadsPerBlock = 256;
  dim3 block{threadsPerBlock / 32, 32, 1};
  dim3 grid{num_blocks(ni, block.x), num_blocks(nj, block.y), 1};
  kernel_dev<<<block, grid>>>(ni, nj, nk, alpha, beta, C, A, B);
}


void run(State &state, pbsize_t pbsize) {
  pbsize_t ni = pbsize - pbsize / 4;
  pbsize_t nj = pbsize - pbsize / 8;
  pbsize_t nk = pbsize;

  real alpha = 1.5;
  real beta = 1.2;
  auto C = state.allocate_array<real>({ni, nj}, /*fakedata*/ true, /*verify*/ true, "C");
  auto A = state.allocate_array<real>({ni, nk}, /*fakedata*/ true, /*verify*/ false, "A");
  auto B = state.allocate_array<real>({nk, nj}, /*fakedata*/ true, /*verify*/ false, "B");


  real *dev_C = state.allocate_dev_hip<real>(ni * nj);
  real *dev_A = state.allocate_dev_hip<real>(ni * nk);
  real *dev_B = state.allocate_dev_hip<real>(nk * nj);

  for (auto &&_ : state) {
    hipMemcpy(dev_C, C.data(), ni * nj * sizeof(real), hipMemcpyHostToDevice);
    hipMemcpy(dev_A, A.data(), ni * nk * sizeof(real), hipMemcpyHostToDevice);
    hipMemcpy(dev_B, B.data(), nk * nj * sizeof(real), hipMemcpyHostToDevice);



    kernel(ni, nj, nk, alpha, beta, dev_C, dev_A, dev_B);


    hipMemcpy(C.data(), dev_C, ni * nj * sizeof(real), hipMemcpyDeviceToHost);

    hipDeviceSynchronize();
  }

  state.free_dev_hip(dev_C);
  state.free_dev_hip(dev_A);
  state.free_dev_hip(dev_B);
}
