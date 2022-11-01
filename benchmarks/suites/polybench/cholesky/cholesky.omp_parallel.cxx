// BUILD: add_benchmark(ppm=omp_parallel)
#include "rosetta.h"
#include <omp.h>


// https://tcevents.chem.uzh.ch/event/3/contributions/16/attachments/14/66/04_Threading_Lab.pdf
// https://www.researchgate.net/figure/OpenMP-based-Cholesky-implementation-in-PLASMA-Handling-of-corner-cases-removed-for_fig12_303980919

static real sqr(real v) {
    return v*v;
}


static void cholesky_base(idx_t base , size_t n,  multarray<real, 2> A) {
  for (idx_t i = 0; i < n; i++) {

    // j<i case
    for (idx_t j = 0; j < i; j++) {
      for (idx_t k = 0; k < j; k++)
        A[base + i][base + j] -= A[base + i][base + k] * A[base+j][base+k];
      A[base+i][base+j] /= A[base+j][base+j];
    }

    // i==j case
    for (idx_t k = 0; k < i; k++)
      A[base+i][base+i] -= sqr(A[base+i][base+k]);
    A[base+i][base+i] = std::sqrt(A[base+i][base+i]);
  }
}




// a[m][m]
// b[n][n]
// x[m][n]
//
// a = A[base_ai + i][base_aj + j]
// b = A[base_bi + i][base_bj + j]
static void sst(idx_t base_ai,idx_t base_aj, 
    idx_t base_bi,idx_t base_bj, 
    pbsize_t m, pbsize_t n, multarray<real, 2> A, multarray<real, 2> X){

#if 0
    for (idx_t j = 0; j < n; ++j){
       // x[(m - 1)*n + j] = b[(m - 1)*n + j] / a[(m - 1) * m + (m - 1)];
        auto Bmj =  A[base_bi + m-1][base_bj + j];
        auto Amm =  A[base_ai + m-1][base_aj + m-1];
        auto Xmj = Bmj / Amm;
        X[m-1][j] = Xmj;
    }
#endif

    for (idx_t i =  0; i < m; ++i){
        //for (idx_t i = m - 2; i >= 0; --i){
        for (idx_t k = 0; k < n; k++){
            real sum = 0;


            for (idx_t j = 0; j < k; ++j){
              //  sum += a[i*m + j] * x[j*n + k];
                auto Aij = A[base_ai + k][base_aj + j];
                auto Xjk = X[j][i];
                sum += Aij * Xjk;
            }

           // x[i*n + k] = (b[i*n + k] - sum) / a[i*m + i];
            auto Aik =  A[base_bi + i][base_bj+k]  ;
            auto sumAik =Aik  - sum;
            auto Aii = A[base_ai+k][base_aj+k];
            auto Xik = sumAik / Aii;
            X[i][k] =Xik;
        }
    }
}










static void my_dtrsm(
    idx_t base_ai,idx_t base_aj, 
    idx_t base_bi,idx_t base_bj, 
    pbsize_t m, pbsize_t n, multarray<real, 2> A, multarray<real, 2> X
  //  const int M, const int N,
   // const real alpha, const real  *A, const int lda, real  *B, const int ldb
){

  //  double* x = (double*)calloc(M*N, sizeof(double));

   // sst(M, N, A, B, x);
    sst(base_ai, base_aj, base_bi, base_bj, m ,n, A, X);

    for (idx_t i = 0; i < m; ++i){
        for (idx_t j = 0; j < n; ++j){
            //B[i*N + j] = x[i*N + j];
            auto Xij = X[i][j];
            A[base_bi + i][base_bj + j] = Xij;
        }
    }
}


static void dtrsm(
    idx_t base_ai,idx_t base_aj, 
    idx_t base_bi,idx_t base_bj, 
    pbsize_t m, pbsize_t n, multarray<real, 2> A, multarray<real, 2> X){
        
    for (idx_t i = 0; i < m; ++i) {
        for (idx_t j = 0; j < m; ++j) {
            auto Bij = A[base_bi + i ][base_bj+j];
            auto Aij = A[base_ai + i ][base_aj+j];

            real sum = 0;
            for (idx_t k = 0; k < m; ++k) {
                auto Bij = A[base_bi + i ][base_bj+k];
                auto Aij = A[base_ai + k ][base_aj+j];
                sum += Aij * Bij;
            }
        }

        auto Bii = A[base_bi + i ][base_bj+i];
        auto Aii = A[base_ai + i ][base_aj+i];
        auto Cii = Bii / Aii  ;
        X[i][i] = Cii ;
    }
}




static void cholesky_gemm(idx_t base_ai, idx_t base_aj, 
    idx_t base_bi, idx_t base_bj,
    idx_t base_ci, idx_t base_cj,
    pbsize_t ni, pbsize_t nj ,pbsize_t nk, multarray<real, 2> A) {
    for (idx_t i = 0; i < ni; ++i) {
        for (idx_t j = 0; j < nj; ++j) {
            for (idx_t k = 0; k < nk; ++k) {
                    assert(base_ai + i >= base_aj + k);
                auto Aik = A[base_ai + i][base_aj + k];  // Li21
                    assert(base_bi + j >= base_bj + k);
                auto Akj = A[base_bi + j][base_bj + k];  // Lj21T
                auto mul  =Aik * Akj;
                    assert(base_ci + i >= base_cj + j);
                A[base_ci + i][base_cj + j] -= mul; // Aij22
            }
        }
    }
}



static void cholesky_syrk(
    idx_t base_ai, idx_t base_aj, 
    idx_t base_bi, idx_t base_bj,
    idx_t base_ci, idx_t base_cj,
    pbsize_t ni, pbsize_t nk ,
    multarray<real, 2> A) {
  for (idx_t i = 0; i < ni; i++) {
    for (idx_t k = 0; k < nk; k++) {
        for (idx_t j = 0; j <= i; j++) {
              assert(base_ai + i >= base_aj + k);
            auto Aik = A[base_ai + i][base_aj + k];
              assert(base_bi + j >= base_bj + k);
            auto Akj = A[base_bi + j][base_bj + k];
            auto mul = Aik * Akj;
              assert(base_ci + i >= base_cj + j);
            A[base_ci + i][base_cj + j] -= mul;
        }
    }
  }
}





static void kernel(pbsize_t n, multarray<real, 2> A,  multarray<real, 2> X) {
    auto *Ap =& A[0][0];
    typedef real ArrTy[3][3];
    ArrTy *Aa = (ArrTy*)( Ap);

    auto suggest_blocksize = []  (idx_t i) -> pbsize_t {
        if (i==0) return 1;
        return 2;
    };




 // #pragma omp parallel firstprivate(n)
  {
      

      for (int ii = 0; ii < n;  ) {
          auto blocksize =  suggest_blocksize(ii);
 
        auto ni = std::min( blocksize, n-ii);



        // Step 1.
         cholesky_base(ii, ni, A);
          

        // Step 2.
      for (int jj = ii+ni; jj < n;  jj+=blocksize) {
          auto nj =   std::min( blocksize, n-jj);
          my_dtrsm(ii, ii, jj, ii, nj, ni, A, X);
      }

        // Step 3.
        for (int jj = ii+ni; jj < n;  jj+=blocksize) {
            auto nj =  std::min( blocksize, n-jj);
            cholesky_syrk(jj, ii,
                          jj, ii,
                          jj, jj, 
                          nj, ni, A );
                  for (int kk = jj+nj; kk < n;  kk+=blocksize) {
                      auto nk =  std::min(blocksize, n-kk);
                        cholesky_gemm(jj,ii, // Li21
                                      kk,ii, // Lj21T
                                      kk,jj, // Aij22
                                      ni,nj,nk,A);
                  }
          }

        ii+=blocksize;
      }
  }
}



// https://math.stackexchange.com/a/358092
static void ensure_posdefinite(int n, multarray<real, 2> A) {
    if (n==3) {
        A[0][0] =  4;
        A[0][1] = 12;
        A[0][2] = -16;
        A[1][0] =  12;
        A[1][1] = 37;
        A[1][2] = -43;
        A[2][0] =  -16;
        A[2][1] = -43;
        A[2][2] = 98;
        return ;
    }

    // make symmetric (not really necessary, the kernel doesn't read the upper triangular elements anyway)
    for (int i = 0; i < n; i++)
        for (int j = 0; j < i; j++) {
            auto val = (std::abs(A[i][j]) + std::abs(A[j][i]))/2;
            A[i][j] = val;
            A[j][i] = val;
        }



    real maximum=0;
    for (int i = 0; i < n ; i++)
        for (int j = 0; j < n; j++) {
            auto val = std::abs(A[i][j]);
            if (val > maximum) maximum = val;
        }


    // Make the diagnonal elements too large to be a linear combination of the other columns (strictly diagonally dominant).
    for (int i = 0; i < n ; i++)
        A[i][i] = std::abs( A[i][i]) + 1 + n*maximum;


    // FIXME: Repeated invocation will grow the numbers
}


void run(State &state, int pbsize) {
    pbsize_t n = pbsize; // 2000

    auto A = state.allocate_array<real>({n, n}, /*fakedata*/ true, /*verify*/ true, "A");
    auto tmp = state.allocate_array<real>({n, n}, /*fakedata*/ false, /*verify*/ false, "tmp"); // TODO: blocksize/allocate locally


    for (auto&& _ : state.manual()) {
        ensure_posdefinite(n, A);
        {
            auto &&scope = _.scope();
            // FIXME: cholesky of pos-definite matrix is not necessarily itself pos-definite
            kernel(n, A, tmp);
        }
    }
}
