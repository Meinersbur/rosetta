// BUILD: add_benchmark(ppm=cuda)

#include <rosetta.h>



static unsigned num_blocks(int num, int factor) {
  return (num + factor - 1) / factor;
}



__global__ void kernel_stencil(pbsize_t n, real A[], real B[]) {
  idx_t i = blockDim.x * blockIdx.x + threadIdx.x + 1;


  if (i < n - 1) {
    B[i] = (A[i - 1] + A[i] + A[i + 1]) / 3;
  }
}



static void kernel(pbsize_t tsteps, pbsize_t n, real A[], real B[]) {
  const unsigned int threadsPerBlock = 256;

  for (idx_t t = 1; t <= tsteps; t++) {
    kernel_stencil<<<threadsPerBlock, num_blocks(n, threadsPerBlock)>>>(n, A, B);
    kernel_stencil<<<threadsPerBlock, num_blocks(n, threadsPerBlock)>>>(n, B, A);
  }
}


void run(State &state, pbsize_t pbsize) {
  pbsize_t tsteps = 1; // 500
  pbsize_t n = pbsize; // 2000



  auto A = state.allocate_array<real>({n}, /*fakedata*/ true, /*verify*/ false, "A");
  auto B = state.allocate_array<real>({n}, /*fakedata*/ false, /*verify*/ true, "B");

  real *dev_A = state.allocate_dev<real>(n);
  real *dev_B = state.allocate_dev<real>(n);

  for (auto &&_ : state) {
    BENCH_CUDA_TRY(cudaMemcpy(dev_A, A.data(), n * sizeof(real), cudaMemcpyHostToDevice));

    kernel(tsteps, n, dev_A, dev_B);

    BENCH_CUDA_TRY(cudaMemcpy(B.data(), dev_B, n * sizeof(real), cudaMemcpyDeviceToHost));

    BENCH_CUDA_TRY(cudaDeviceSynchronize());
  }

  state.free_dev(dev_A);
  state.free_dev(dev_B);
}
