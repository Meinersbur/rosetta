// BUILD: add_benchmark(ppm=cuda)
#include "rosetta.h"





static
unsigned num_blocks(int num, int factor) {
    return (num + factor -1)/factor ;
}



__global__ void kernel_q(pbsize_t m, pbsize_t n,real* A, real s[], real q[], real p[], real r[]) {
    idx_t i = blockDim.x * blockIdx.x + threadIdx.x;

if (i< n) {
      q[i] = 0;
    for (idx_t j = 0; j < m; j++)
      q[i] += A[i*m+j] * p[j];
}
}


__global__ void kernel_s(pbsize_t m, pbsize_t n,real* A, real s[], real q[], real p[], real r[]) {
    idx_t j = blockDim.x * blockIdx.x + threadIdx.x;

if (j< m) {
                  s[j] = 0;
      for (idx_t i = 0; i < n; i++)
        s[j] += r[i] * A[i*m+j];
}
}





static void kernel(pbsize_t m, pbsize_t n, real * A, real s[], real q[], real p[], real r[]) {
  unsigned threadsPerBlock = 256;
  kernel_q<<<threadsPerBlock, num_blocks(n,threadsPerBlock)>>>(m,n,A,s,q,p,r);
  kernel_s<<<threadsPerBlock, num_blocks(m,threadsPerBlock)>>>(m,n,A,s,q,p,r);
}


void run(State &state, pbsize_t pbsize) {
  pbsize_t m = pbsize - 19 * pbsize / 21; // 1900
  pbsize_t n = pbsize;                    // 2100

  auto A = state.allocate_array<real>({n, m}, /*fakedata*/ true, /*verify*/ false, "A");
  auto s = state.allocate_array<real>({m}, /*fakedata*/ false, /*verify*/ true, "s");
  auto q = state.allocate_array<real>({n}, /*fakedata*/ false, /*verify*/ true, "q");
  auto p = state.allocate_array<real>({m}, /*fakedata*/ true, /*verify*/ false, "p");
  auto r = state.allocate_array<real>({n}, /*fakedata*/ true, /*verify*/ false, "r");

  real* dev_A  = state.allocate_dev<real>(n*m);
  real* dev_s  = state.allocate_dev<real>(m);
    real* dev_q  = state.allocate_dev<real>(n);
      real* dev_p  = state.allocate_dev<real>(m);
        real* dev_r  = state.allocate_dev<real>(n);
 

  for (auto &&_ : state) {
    cudaMemcpy(dev_A, A.data(), n*m * sizeof(real), cudaMemcpyHostToDevice);
       //    cudaMemset(dev_s, '\0', m * sizeof(real));
      //            cudaMemset(dev_q, '\0', n * sizeof(real));
      cudaMemcpy(dev_p, p.data(), m * sizeof(real), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_r, r.data(), n* sizeof(real), cudaMemcpyHostToDevice);

    kernel(m, n, dev_A, dev_s, dev_q, dev_p, dev_r);

      cudaMemcpy( s.data() ,dev_s,  m* sizeof(real), cudaMemcpyDeviceToHost ); 
            cudaMemcpy( q.data() ,dev_q,  n* sizeof(real), cudaMemcpyDeviceToHost ); 

     cudaDeviceSynchronize();
  }

     state.free_dev(dev_A);
         state.free_dev(dev_s);
             state.free_dev(dev_q);
                 state.free_dev(dev_p);
                     state.free_dev(dev_r);
}
