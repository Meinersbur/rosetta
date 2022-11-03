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
                auto Aij = A[base_ai + k][base_aj + j]; assert(base_ai + k >=base_aj + j); 
                auto Xjk = X[i][j];
                sum += Aij * Xjk;
            }

            // x[i*n + k] = (b[i*n + k] - sum) / a[i*m + i];
            auto Aik =  A[base_bi + i][base_bj+k]; assert(base_bi + i >= base_bj+k);
            auto sumAik =Aik  - sum;
            auto Aii = A[base_ai+k][base_aj+k]; assert(base_ai+k >= base_bj+k);
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
            auto Xij = X[i][j]; assert(base_bi + i >= base_bj + j);
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
    auto *Ap =& A[0][0];
    typedef real ArrTy[6][6];
    ArrTy *Aa = (ArrTy*)( Ap);

    for (idx_t i = 0; i < ni; ++i) {
        for (idx_t j = 0; j < nj; ++j) {
            for (idx_t k = 0; k < nk; ++k) {
                    assert(base_ai + i >= base_aj + k);
                auto Aik = A[base_ai + i][base_aj + k];  // A = Li21
                    assert(base_bi + j >= base_bj + k);
                auto Akj = A[base_bi + j][base_bj + k];  // B = Lj21T
                auto mul = Aik * Akj;
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
    auto *Ap =& A[0][0];
    typedef real ArrTy[6][6];
    ArrTy *Aa = (ArrTy*)( Ap);

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
    typedef real ArrTy[6][6];
    ArrTy *Aa = (ArrTy*)( Ap);

    auto suggest_blocksize = []  (idx_t i) -> pbsize_t {
        return 2;
        if (i==0) return 1;
        return 3;
    };




 // #pragma omp parallel firstprivate(n)
  {
      

      for (int ii = 0; ii < n;  ) {
          auto blocksize =  suggest_blocksize(ii);
 
        auto ni = std::min( blocksize, n-ii);



        // Step 1.
         cholesky_base(ii, ni, A); // A[ii..ii+n][ii..ii+n] <- cholesky(A[ii..ii+n][ii..ii+n])
          

        // Step 2.
      for (int jj = ii+ni; jj < n;  jj+=blocksize) {
          auto nj =   std::min( blocksize, n-jj);
          my_dtrsm(ii, ii, jj, ii, nj, ni, A, X); // A[jj..jj+n][ii..ii+n] <- A[jj..jj+n][ii..ii+n] * A[ii..ii+n][ii..ii+n]^-1
                                                  //          B            <-        B                        A^-1
                                                  //        L21_i          <-       A21_i           *        L11_T^1 
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

                      // A[kk..kk+nk][jj..jj+nj] <- A[kk..kk+nk][jj..jj+nj] - A[jj..jj+nj][ii..ii+ni] * A[kk..kk+nj][ii..ii+ni]^T (UR space)
                      // A[jj..jj+nj][kk..kk+nk] <- A[jj..jj+nj][kk..kk+nk] - A[ii..ii+ni][jj..jj+nj] * A[ii..ii+ni][kk..kk+nj]^T (LL space)
                      //          C              <-         C               -   A   *   B
                      //       A22_j,k           <-      A22_j,k            - L21_j * L21_k^T
                      //       A22_i,j           <-      A22_i,j            - L21_i * L21_j^T
                      cholesky_gemm(kk,ii, // A = A[ii..ii+ni][jj..jj+nj]   = L21_i
                                    jj,ii, // B = A[ii..ii+ni][kk..kk+nj]^T = L21_j^T
                                    kk,jj, // C = A[jj..jj+nj][kk..kk+nk]   = A22_i,j
                                    ni,nj,nk,A);
                  }
          }

        ii+=blocksize;
      }
  }
}



// https://math.stackexchange.com/a/358092
static void ensure_posdefinite(pbsize_t n, multarray<real, 2> A) {
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
    } else if (n==4) {
        real B[4][4] = {0};
        B[0][0] =  1;
        B[1][0] =   2;
        B[2][0] =   3;
        B[3][0] =  4;
        B[1][1] =    5;
        B[2][1] =    6;
        B[3][1] =    7;
        B[2][2] =      8;
        B[3][2] =     9;
        B[3][3] =      10;

        for (idx_t i = 0;i < n; ++i )
            for (idx_t j = 0;j < n; ++j )
                A[i][j] = 0;
 
        for (idx_t i = 0;i < n; ++i )
            for (idx_t j = 0;j < n; ++j )
                for (idx_t k = 0;k < n; ++k )
                    A[i][j] += B[i][k] * B[j][k];
        return ;
    } else if (n==5) {
        real B[5][5] = {0};
        int k = 1;
        for (idx_t i = 0;i < n; ++i )
            for (idx_t j = 0; j <=i; ++j ) {
                B[i][j] = k;
                k+=1;
            }



        for (idx_t i = 0;i < n; ++i )
            for (idx_t j = 0;j < n; ++j )
                A[i][j] = 0;

        for (idx_t i = 0;i < n; ++i )
            for (idx_t j = 0;j < n; ++j )
                for (idx_t k = 0;k < n; ++k )
                    A[i][j] += B[i][k] * B[j][k];
        return ;
    } else if (n==6) {
        real B[6][6] = {0};
        int k = 1;
        for (idx_t i = 0;i < n; ++i )
            for (idx_t j = 0; j < n; ++j ) 
                if (j >= i) {
                    B[j][i] = k;
                    k += 1;
                }
            



        for (idx_t i = 0;i < n; ++i )
            for (idx_t j = 0;j < n; ++j )
                A[i][j] = 0;

        for (idx_t i = 0;i < n; ++i )
            for (idx_t j = 0;j < n; ++j )
                for (idx_t k = 0;k < n; ++k )
                    A[i][j] += B[i][k] * B[j][k];
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
