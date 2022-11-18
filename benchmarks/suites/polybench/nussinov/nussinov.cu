// BUILD: add_benchmark(ppm=cuda)

#include <rosetta.h>
#include <cuda.h>
#include <cuda_runtime.h>


static
unsigned num_blocks(int num, int factor) {
    return (num + factor -1)/factor ;
}


template<typename T>
struct AtomicMax;

template<>
struct AtomicMax<double> {
    __device__  static double set_if_larger(double &dst, double val) {
    // Type-prune everything as uint64_t because there is no floating-point version of atomicCAS.
    unsigned long long int*dstptr = (unsigned long long int*)&dst;
        unsigned long long int newval = __double_as_longlong(val);
        unsigned long long int dstval = *dstptr;
        while (1) {
            // Values can only get larger.
            // Special attention for NaN where every comparison is False. If dstval is already NaN, we are done. Otherwise, set it to NaN and come back here.
            if (__longlong_as_double(dstval) >= __longlong_as_double(newval) || isnan( __longlong_as_double(dstval))) return;

            auto assumed = dstval;
            dstval = atomicCAS(dstptr, assumed, newval);

            // Three possibilities:
            // 1. Noone interferred and we set the new max value.
            // if (assumed == dstval) return;

            // 2. Someone else overwrote dst with a value between newval and assumed.
            // Will continue the loop again, same problem except that dst now contains dstval.

            // 3. Someone else overwrote dst with a larger value than newval.
            // dstval contains that largest value.
            // Will break the loop at next iteration because dstval >= newval.
        }
    }
};







__global__ void kernel_max_score(pbsize_t n, real seq[], real * table, idx_t i) {
idx_t j  = blockDim.x * blockIdx.x + threadIdx.x + i+1;


    if (i < n) {
            real maximum =  table[i*n+j];

                if (j - 1 >= 0)
                maximum = max(maximum, table[i*n+j - 1]);
            if (i + 1 < n)
                 maximum = max(maximum, table[(i + 1)*n+j]);

            if (j - 1 >= 0 && i + 1 < n) {
                /* don't allow adjacent elements to bond */
                if (i < j - 1)
                    maximum= max(maximum, table[(i + 1)*n+j - 1] + (seq[i] +  seq[j] == 3) ? (real)1 : (real)0);
                else
                    maximum = max(maximum, table[(i + 1)*n+j - 1]);
            }

            for (idx_t k = i + 1; k < j; k++)
                maximum = max(maximum, table[i*n+k] + table[(k + 1)*n+j]);
            AtomicMax<real>:: set_if_larger(   table[i*n+j] , maximum);
    }
}



static void kernel(pbsize_t n, real seq[], real * table) {
    const  unsigned threadsPerBlock = 256;

    for (idx_t i = n - 1; i >= 0; i--) {
        kernel_max_score<<<threadsPerBlock,num_blocks(n-(i+1),threadsPerBlock)>>>(n,seq,table,i);
    }

#if 0
    for (idx_t i = n - 1; i >= 0; i--) {
        for (idx_t j = i + 1; j < n; j++) {
            if (j - 1 >= 0)
                table[i][j] = max_score(table[i][j], table[i][j - 1]);
            if (i + 1 < n)
                table[i][j] = max_score(table[i][j], table[i + 1][j]);

            if (j - 1 >= 0 && i + 1 < n) {
                /* don't allow adjacent elements to bond */
                if (i < j - 1)
                    table[i][j] = max_score(table[i][j], table[i + 1][j - 1] + match(seq[i], seq[j]));
                else
                    table[i][j] = max_score(table[i][j], table[i + 1][j - 1]);
            }

            real maximum = table[i][j] ;
#pragma omp parallel for default(none) firstprivate(i,j,n,table) schedule(static) reduction(max:maximum)
            for (idx_t k = i + 1; k < j; k++)
                maximum = max_score(maximum, table[i][k] + table[k + 1][j]);
            table[i][j] = maximum;
        }
    }
    #endif
}



void run(State &state, pbsize_t pbsize) {
    pbsize_t n = pbsize; // 2500



    auto seq = state.allocate_array<real>({n}, /*fakedata*/ true, /*verify*/ false, "seq");
    auto table = state.allocate_array<real>({n, n}, /*fakedata*/ true, /*verify*/ true, "table");


  real* dev_seq = state.allocate_dev<real>(n);
  real* dev_table = state.allocate_dev<real>(n*n);


  for (auto &&_ : state) {
    BENCH_CUDA_TRY( cudaMemcpy(dev_seq, seq.data(),  n* sizeof(real), cudaMemcpyHostToDevice));
    BENCH_CUDA_TRY( cudaMemcpy(dev_table, table.data(),  n*n* sizeof(real), cudaMemcpyHostToDevice));
    kernel(n, dev_seq, dev_table);
    BENCH_CUDA_TRY(    cudaMemcpy( table.data() ,dev_table,  n*n*sizeof(real), cudaMemcpyDeviceToHost ));

    BENCH_CUDA_TRY(  cudaDeviceSynchronize());
  }

  state.free_dev(dev_seq);
  state.free_dev(dev_table);
}
