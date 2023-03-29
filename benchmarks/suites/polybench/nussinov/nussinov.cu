// BUILD: add_benchmark(ppm=cuda)

#include <cuda.h>
#include <cuda_runtime.h>
#include <rosetta.h>


static unsigned num_blocks(int num, int factor) {
  return (num + factor - 1) / factor;
}


template <typename T>
struct AtomicMax;

template <>
struct AtomicMax<double> {
  __device__ static double set_if_larger(double &dst, double val) {

    // Type-prune everything as uint64_t because there is no floating-point version of atomicCAS.
    unsigned long long int *dstptr = (unsigned long long int *)&dst;
    unsigned long long int newval = __double_as_longlong(val);
    unsigned long long int dstval = *dstptr;

    while (1) {
        // Special attention for NaN where every comparison is False. If dstval is already NaN, we are done. Otherwise, set it to newval (which can be NaN) and come back here.
        if (isnan(__longlong_as_double(dstval)))
            return;

      // Values can only get larger.
      if (__longlong_as_double(dstval) >= __longlong_as_double(newval) )
        return;

      auto assumed = dstval;
      dstval = atomicCAS(dstptr, assumed, newval);

      // Three possibilities after atomicCAS:

      // 1. Noone interfered and we set the new max value.
      // if (assumed == dstval) return;
      // Included in the dstval >= newval test in the next iteration

      // 2. Someone else overwrote dstptr with a value between newval and assumed.
      // Will continue the loop again, same problem except that dstptr now contains dstval.

      // 3. Someone else overwrote dst with a larger value than newval.
      // dstval contains that largest value.
      // Will break the loop at next iteration because dstval >= newval.
    }
  }
};




// Dynamic programming wavefront
__global__ void kernel_max_score(pbsize_t n, real seq[], real table[], real oldtable[], idx_t w) {
  idx_t j = blockDim.x * blockIdx.x + threadIdx.x;
  idx_t i = ((idx_t)n-1) +j-w; 

  if (0 <= i && i < n && i+1 <= j && j < n) {
    real maximum = table[i * n + j];

    if (j - 1 >= 0)
      maximum = max(maximum, table[i * n + (j - 1)]);
    if (i + 1 < n)
      maximum = max(maximum, table[(i + 1) * n + j]);

    if (j - 1 >= 0 && i + 1 < n) {
        auto upd = table[(i + 1) * n + (j - 1)];

      /* don't allow adjacent elements to bond */
        if (i < j - 1)
            upd += (seq[i] + seq[j] == 3) ? (real)1 : (real)0;

        maximum = max(maximum, upd);
    }

    for (idx_t k = i + 1; k < j; k++)
      maximum = max(maximum, table[i * n + k] + table[(k + 1) * n + j]);

  //  AtomicMax<real>::set_if_larger(table[i * n + j], maximum);
    table[i * n + j] = maximum;
  }
}



static void kernel(pbsize_t n, real seq[], real table[],  real oldtable[]) {
  const unsigned threadsPerBlock = 32;

  for (idx_t w = n; w < 2*n-1; ++w) { // wavefronting 
    kernel_max_score<<<num_blocks(n,threadsPerBlock), threadsPerBlock>>>(n, seq, table,oldtable, w);
  }
}



void run(State &state, pbsize_t pbsize) {
  pbsize_t n = pbsize; // 2500



  auto seq = state.allocate_array<real>({n}, /*fakedata*/ true, /*verify*/ false, "seq");
  auto table = state.allocate_array<real>({n, n}, /*fakedata*/ true, /*verify*/ true, "table");

  real *dev_seq = state.allocate_dev<real>(n);
  real *dev_table = state.allocate_dev<real>(n * n);
  real *dev_oldtable = state.allocate_dev<real>(n * n);


  for (auto &&_ : state) {
    BENCH_CUDA_TRY(cudaMemcpy(dev_seq, seq.data(), n * sizeof(real), cudaMemcpyHostToDevice));
    BENCH_CUDA_TRY(cudaMemcpy(dev_table, table.data(), n * n * sizeof(real), cudaMemcpyHostToDevice));
    BENCH_CUDA_TRY(cudaMemcpy(dev_oldtable, dev_table, n * n * sizeof(real), cudaMemcpyDeviceToDevice));
    kernel(n, dev_seq, dev_table, dev_oldtable );
    BENCH_CUDA_TRY(cudaMemcpy(table.data(), dev_table, n * n * sizeof(real), cudaMemcpyDeviceToHost));

    BENCH_CUDA_TRY(cudaDeviceSynchronize());
  }

  state.free_dev(dev_seq);
  state.free_dev(dev_table);
  state.free_dev(dev_oldtable);
}
