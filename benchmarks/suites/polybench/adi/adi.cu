// BUILD: add_benchmark(ppm=cuda)
#include "rosetta.h"



__global__ void kernel_column_sweep(pbsize_t tsteps,
    pbsize_t n,
 real* u,
 real*v,
 real* p,
 real* q, real a,  real b, real c, real d, real e, real f) {
    idx_t i = blockDim.x * blockIdx.x + threadIdx.x + 1;
 

if (i < n-1) {
      v[0*n+i] = 1;
      p[i*n+0] = 0;
      q[i*n+0] = v[0*n+i];
      for (idx_t j = 1; j < n - 1; j++) {
        p[i*n+j] = -c / (a * p[i*n+ j - 1] + b);
        q[i*n+j] = (-d * u[j*n+i - 1] + (1 + 2 * d) * u[j*n+i] - f * u[j*n+ i + 1] - a * q[i*n+j - 1]) / (a * p[i*n+j - 1] + b);
      }

      v[(n - 1)*n+i] = 1;
      for (idx_t j = n - 2; j >= 1; j--)
        v[j*n+i] = p[i*n+j] * v[(j + 1)*n+i] + q[i*n+j];
    }
}


__global__ void kernel_row_sweep(pbsize_t tsteps, pbsize_t n, real* u, real*v, real* p, real* q, real a,  real b, real c, real d, real e, real f) {
    idx_t i = blockDim.x * blockIdx.x + threadIdx.x + 1;
 
if (i < n-1) {
      u[i*n+0] = 1;
      p[i+n+0] = 0;
      q[i*n+0] = u[i*n+0];
      for (idx_t j = 1; j < n - 1; j++) {
        p[i*n+j] = -f / (d * p[i*n+j - 1] + e);
        q[i*n+j] = (-a * v[(i - 1)*n+j] + (1 + 2 * a) * v[i*n+j] - c * v[(i + 1)*n+j] - d * q[i*n+j - 1]) / (d * p[i*n+j - 1] + e);
      }
      u[i*n+n - 1] = 1;
      for (idx_t j = n - 2; j >= 1; j--)
        u[i*n+j] = p[i*n+j] * u[i*n+j + 1] + q[i*n+j];
    }
}


static
unsigned num_blocks(int num, int factor) {
    return (num + factor -1)/factor ;
}



static void kernel(
    pbsize_t tsteps,
    pbsize_t n,
 real* u,
 real*v,
 real* p,
 real* q) {
    unsigned threadsPerBlock = 256;

  real DX = 1 / (real)n;
  real DY = 1 / (real)n;
  real DT = 1 / (real)tsteps;
  real B1 = 2;
  real B2 = 1;
  real mul1 = B1 * DT / (DX * DX);
  real mul2 = B2 * DT / (DY * DY);

  real a = -mul1 / 2;
  real b = 1 + mul1;
  real c = a;
  real d = -mul2 / 2;
  real e = 1 + mul2;
  real f = d;



  for (idx_t t = 1; t <= tsteps; t++) {
    // Column Sweep
    kernel_column_sweep <<<threadsPerBlock ,num_blocks(n-2,threadsPerBlock) >>> (tsteps, n,u,v,p,q,a,b,c,d,e,f);
    
    // Row Sweep
    kernel_row_sweep <<<threadsPerBlock ,num_blocks(n-2,threadsPerBlock) >>> (tsteps, n,u,v,p,q,a,b,c,d,e,f);
  }
}



void run(State &state, pbsize_t pbsize) {
  pbsize_t tsteps = pbsize / 2; // 500
  pbsize_t n = pbsize;          // 1000



  auto u = state.allocate_array<double>({n, n}, /*fakedata*/ true, /*verify*/ true,"u");
  //auto v = state.allocate_array<double>({n, n}, /*fakedata*/ false, /*verify*/ false,"v");
 // auto p = state.allocate_array<double>({n, n}, /*fakedata*/ false, /*verify*/ false, "p");
 // auto q = state.allocate_array<double>({n, n}, /*fakedata*/ false, /*verify*/ false, "q");


 real* dev_u  = state.allocate_dev<real>(n* n);
 real* dev_v  = state.allocate_dev<real>(n* n);
  real* dev_p  = state.allocate_dev<real>(n* n);
   real* dev_q  = state.allocate_dev<real>(n* n);

  for (auto &&_ : state) {
      cudaMemcpy(dev_u, u.data(),n*n* sizeof(real), cudaMemcpyHostToDevice);
                   cudaMemset(dev_v, '\0', n*n * sizeof(real) );
                                cudaMemset(dev_p, '\0',  n*n * sizeof(real) );
              cudaMemset(dev_q, '\0', n*n * sizeof(real) );
                                             


      kernel(tsteps, n, dev_u, dev_v, dev_p, dev_q);
    

          cudaMemcpy( u.data() ,dev_u,  n*n * sizeof(real), cudaMemcpyDeviceToHost ); 

          cudaDeviceSynchronize();
  }

         state.free_dev(dev_u);
              state.free_dev(dev_v);
                   state.free_dev(dev_p);
                        state.free_dev(dev_q);
}


