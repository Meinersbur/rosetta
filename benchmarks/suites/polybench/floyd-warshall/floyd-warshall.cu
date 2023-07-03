// BUILD: add_benchmark(ppm=cuda)

#include <cuda.h>
#include <rosetta.h>

static unsigned num_blocks(int num, int factor) {
  return (num + factor - 1) / factor;
}



template <typename T>
struct AtomicMin;

template <>
struct AtomicMin<double> {
  __device__ static double set_if_smaller(double &dst, double val) {
    // Store everything as uint64_t to protect from NaNs.
    unsigned long long int newval = __double_as_longlong(val);
    unsigned long long int old = *((unsigned long long int *)&dst);
    while (1) {
      // Values can only get smaller
      if (__longlong_as_double(old) <= __longlong_as_double(newval))
        return __longlong_as_double(old);

      auto assumed = old;
      auto newold = atomicCAS((unsigned long long int *)&dst, assumed, newval);

      // Three possibilities:
      // 1. Noone interferred and we set the new min value, even if it was NaN.
      if (assumed == newold)
        return __longlong_as_double(old);

      // 2. Someone else overwrote dst with a value between val and old.
      // Will continue the loop again, same problem except that dst now contains old
      old = newold;

      // 3. Someone else overwrote dst with a smaller value than val.
      // dst and old both contains that smallest value
      // Will break the loop at next iteration because old <= newval
    }
  }
};



__global__ void kernel_min(pbsize_t n, real *path, idx_t k) {
  idx_t i = blockDim.x * blockIdx.x + threadIdx.x;
  idx_t j = blockDim.y * blockIdx.y + threadIdx.y;


  if (i < n && j < n)
    AtomicMin<real>::set_if_smaller(path[i * n + j], path[i * n + k] + path[k * n + j]);
}



static void kernel(pbsize_t n, real *path) {
  const unsigned threadsPerBlock = 256;

  for (idx_t k = 0; k < n; k++) {
    dim3 block{threadsPerBlock / 32, 32, 1};
    dim3 grid{num_blocks(n, block.x), num_blocks(n, block.y), 1};
    kernel_min<<<block, grid>>>(n, path, k);
  }
}



void run(State &state, pbsize_t pbsize) {
  pbsize_t n = pbsize; // 2800



  auto path = state.allocate_array<real>({n, n}, /*fakedata*/ true, /*verify*/ true, "path");

  real *dev_path = state.allocate_dev<real>(n * n);

  for (auto &&_ : state) {
    BENCH_CUDA_TRY(cudaMemcpy(dev_path, path.data(), n * n * sizeof(real), cudaMemcpyHostToDevice));

    kernel(n, dev_path);


    BENCH_CUDA_TRY(cudaMemcpy(path.data(), dev_path, n * n * sizeof(real), cudaMemcpyDeviceToHost));

    BENCH_CUDA_TRY(cudaDeviceSynchronize());
  }

  state.free_dev(dev_path);
}
