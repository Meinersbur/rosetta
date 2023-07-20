// BUILD: add_benchmark(hip,
// BUILD:               GenParam('real',compiletime,choices=['float','double','long double']),
// BUILD:               SizeParam('n',runtime,verify=129,train=1024,ref=1024*1024,min=0),
// BUILD:               TuneParam('threadsPerBlock',runtime)
// BUILD:              )

#include <rosetta.h>

__global__ void hip_assign(pbsize_t n, real *data) {
  idx_t i = blockDim.x * blockIdx.x + threadIdx.x;

  if (i < n)
    data[i] = i;
}


static int num_blocks(int num, int factor) {
  return (num + factor - 1) / factor;
}


void run(State &state, pbsize_t pbsize) {
  pbsize_t n = pbsize;


  auto data = state.allocate_array<real>({n}, /*fakedata*/ false, /*verify*/ true, "data");
  real *dev_data = state.allocate_dev_hip<real>(n);

  for (auto &&_ : state) {
    const int threadsPerBlock = 256;
    hip_assign<<<num_blocks(n, threadsPerBlock), threadsPerBlock>>>(n, dev_data);

    BENCH_HIP_TRY(hipMemcpy(data.data(), dev_data, n * sizeof(real), hipMemcpyDeviceToHost));
    BENCH_HIP_TRY(hipDeviceSynchronize());
  }

  state.free_dev_hip(dev_data);
}
