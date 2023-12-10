// BUILD: add_benchmark(ppm=hip)

#include <rosetta.h>

static unsigned num_blocks(int num, int factor) {
  return (num + factor - 1) / factor;
}



__global__ void kernel_stencil(pbsize_t tsteps, pbsize_t n,
                               real *A) {
  idx_t i = blockDim.x * blockIdx.x + threadIdx.x + 1;
  idx_t j = blockDim.y * blockIdx.y + threadIdx.y + 1;


  if (i < n - 1 && j < n - 1)
    A[i * n + j] = (A[(i - 1) * n + j - 1] +
                    A[(i - 1) * n + j] +
                    A[(i - 1) * n + j + 1] +
                    A[i * n + j - 1] +
                    A[i * n + j] +
                    A[i * n + j + 1] +
                    A[(i + 1) * n + j - 1] +
                    A[(i + 1) * n + j] +
                    A[(i + 1) * n + j + 1]) /
                   9;
}



static void kernel(pbsize_t tsteps, pbsize_t n,
                   real *A) {
  // FIXME: Parallelizing this should give different results
  const unsigned int threadsPerBlock = 256;

  for (idx_t t = 1; t <= tsteps; t++) {
    dim3 block{threadsPerBlock / 32, 32, 1};
    dim3 grid{num_blocks(n - 2, block.x), num_blocks(n - 2, block.y), 1};
    kernel_stencil<<<block, grid>>>(tsteps, n, A);
  }
}


void run(State &state, int pbsize) {
  pbsize_t tsteps = 1; // 500
  pbsize_t n = pbsize; // 2000


  // Changed verify to false. Result is non-deterministic by the algorithm by design
  auto A = state.allocate_array<real>({n, n}, /*fakedata*/ true, /*verify*/ false, "A");

  real *dev_A = state.allocate_dev_hip<real>(n * n);

  for (auto &&_ : state) {
    BENCH_HIP_TRY(hipMemcpy(dev_A, A.data(), n * n * sizeof(real), hipMemcpyHostToDevice));
    kernel(tsteps, n, dev_A);
    BENCH_HIP_TRY(hipMemcpy(A.data(), dev_A, n * n * sizeof(real), hipMemcpyDeviceToHost));

    BENCH_HIP_TRY(hipDeviceSynchronize());
  }

  state.free_dev_hip(dev_A);
}
