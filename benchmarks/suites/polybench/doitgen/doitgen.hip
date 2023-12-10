// BUILD: add_benchmark(ppm=hip)

#include "rosetta.h"

static unsigned num_blocks(int num, int factor) {
  return (num + factor - 1) / factor;
}


__global__ void kernel_sum(pbsize_t nr, pbsize_t nq, pbsize_t np,
                           real *A,
                           real *C4, real *sum) {
  idx_t r = blockDim.x * blockIdx.x + threadIdx.x;
  idx_t q = blockDim.y * blockIdx.y + threadIdx.y;
  idx_t p = blockDim.z * blockIdx.z + threadIdx.z;


  if (r < nr && q < nq && p < np) {
    sum[(r * nq + q) * np + p] = 0;
    for (idx_t s = 0; s < np; s++)
      sum[(r * nq + q) * np + p] += A[(r * nq + q) * np + s] * C4[s * np + p];
  }
}



static void kernel(pbsize_t nr, pbsize_t nq, pbsize_t np,
                   real *A,
                   real *C4, real *sum) {
  const unsigned threadsPerBlock = 256;

  dim3 block{1, threadsPerBlock / 32, 32};
  dim3 grid{num_blocks(nr, block.x), num_blocks(nq, block.y), num_blocks(np, block.z)};
  kernel_sum<<<block, grid>>>(nr, nq, np, A, C4, sum);
}



void run(State &state, pbsize_t pbsize) {
  pbsize_t nq = pbsize - pbsize / 8;  // 140
  pbsize_t nr = pbsize - pbsize / 16; // 150
  pbsize_t np = pbsize;               // 160



  auto A = state.allocate_array<real>({nr, nq, np}, /*fakedata*/ true, /*verify*/ true, "A");
  auto C4 = state.allocate_array<real>({np, np}, /*fakedata*/ true, /*verify*/ false, "C4");


  real *dev_A = state.allocate_dev_hip<real>(nr * nq * np);
  real *dev_C4 = state.allocate_dev_hip<real>(np * np);
  real *dev_sum = state.allocate_dev_hip<real>(nr * nq * np);

  for (auto &&_ : state) {
    BENCH_HIP_TRY(hipMemcpy(dev_A, A.data(), nr * nq * np * sizeof(real), hipMemcpyHostToDevice));
    BENCH_HIP_TRY(hipMemcpy(dev_C4, C4.data(), np * np * sizeof(real), hipMemcpyHostToDevice));

    kernel(nr, nq, np, dev_A, dev_C4, dev_sum);

    hipMemcpy(A.data(), dev_sum, nr * nq * np * sizeof(real), hipMemcpyDeviceToHost);

    BENCH_HIP_TRY(hipDeviceSynchronize());
  }

  state.free_dev_hip(dev_A);
  state.free_dev_hip(dev_C4);
  state.free_dev_hip(dev_sum);
}
