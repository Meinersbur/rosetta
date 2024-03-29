// BUILD: add_benchmark(ppm=omp_task,sources=[__file__, "cholesky-common.cxx"])
#include "cholesky-common.h"
#include <omp.h>
#include <rosetta.h>

// https://tcevents.chem.uzh.ch/event/3/contributions/16/attachments/14/66/04_Threading_Lab.pdf
// https://www.researchgate.net/figure/OpenMP-based-Cholesky-implementation-in-PLASMA-Handling-of-corner-cases-removed-for_fig12_303980919
// https://www.cs.utexas.edu/users/flame/Notes/NotesOnChol.pdf


static real sqr(real v) {
  return v * v;
}


static void cholesky_base(idx_t base, size_t n, multarray<real, 2> A) {
  for (idx_t i = 0; i < n; i++) {

    // j<i case
    for (idx_t j = 0; j < i; j++) {
      for (idx_t k = 0; k < j; k++)
        A[base + i][base + j] -= A[base + i][base + k] * A[base + j][base + k];
      A[base + i][base + j] /= A[base + j][base + j];
    }

    // i==j case
    for (idx_t k = 0; k < i; k++)
      A[base + i][base + i] -= sqr(A[base + i][base + k]);
    A[base + i][base + i] = std::sqrt(A[base + i][base + i]);
  }
}



// a[m][m]
// b[n][n]
// x[m][n]
//
// a = A[base_ai + i][base_aj + j]
// b = A[base_bi + i][base_bj + j]
static void sst(
    idx_t base_ai, idx_t base_aj,
    idx_t base_bi, idx_t base_bj,
    pbsize_t m, pbsize_t n, multarray<real, 2> A, multarray<real, 2> X) {

#if 0
    for (idx_t j = 0; j < n; ++j){
       // x[(m - 1)*n + j] = b[(m - 1)*n + j] / a[(m - 1) * m + (m - 1)];
        auto Bmj =  A[base_bi + m-1][base_bj + j];
        auto Amm =  A[base_ai + m-1][base_aj + m-1];
        auto Xmj = Bmj / Amm;
        X[m-1][j] = Xmj;
    }
#endif

  for (idx_t i = 0; i < m; ++i) {
    // for (idx_t i = m - 2; i >= 0; --i){
    for (idx_t k = 0; k < n; k++) {
      real sum = 0;


      for (idx_t j = 0; j < k; ++j) {
        //  sum += a[i*m + j] * x[j*n + k];
        auto Aij = A[base_ai + k][base_aj + j];
        assert(base_ai + k >= base_aj + j);
        auto Xjk = X[base_bi + i][j];
        sum += Aij * Xjk;
      }

      // x[i*n + k] = (b[i*n + k] - sum) / a[i*m + i];
      auto Aik = A[base_bi + i][base_bj + k];
      assert(base_bi + i >= base_bj + k);
      auto sumAik = Aik - sum;
      auto Aii = A[base_ai + k][base_aj + k];
      assert(base_ai + k >= base_bj + k);
      auto Xik = sumAik / Aii;
      X[base_bi + i][k] = Xik;
    }
  }
}

static void my_dtrsm(
    idx_t base_ai, idx_t base_aj,
    idx_t base_bi, idx_t base_bj,
    pbsize_t m, pbsize_t n, multarray<real, 2> A, multarray<real, 2> X
    //  const int M, const int N,
    // const real alpha, const real  *A, const int lda, real  *B, const int ldb
) {

  //  double* x = (double*)calloc(M*N, sizeof(double));

  // sst(M, N, A, B, x);
  sst(base_ai, base_aj, base_bi, base_bj, m, n, A, X);

  for (idx_t i = 0; i < m; ++i) {
    for (idx_t j = 0; j < n; ++j) {
      // B[i*N + j] = x[i*N + j];
      auto Xij = X[base_bi + i][j];

      assert(base_bi + i >= base_bj + j);
      A[base_bi + i][base_bj + j] = Xij;
    }
  }
}



static void cholesky_gemm(idx_t base_ai, idx_t base_aj,
                          idx_t base_bi, idx_t base_bj,
                          idx_t base_ci, idx_t base_cj,
                          pbsize_t ni, pbsize_t nj, pbsize_t nk, multarray<real, 2> A) {
  auto *Ap = &A[0][0];
  typedef real ArrTy[6][6];
  ArrTy *Aa = (ArrTy *)(Ap);

  for (idx_t i = 0; i < ni; ++i) {
    for (idx_t j = 0; j < nj; ++j) {
      for (idx_t k = 0; k < nk; ++k) {
        assert(base_ai + i >= base_aj + k);
        auto Aik = A[base_ai + i][base_aj + k]; // A = Li21
        assert(base_bi + j >= base_bj + k);
        auto Akj = A[base_bi + j][base_bj + k]; // B = Lj21T
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
    pbsize_t ni, pbsize_t nk,
    multarray<real, 2> A) {
  auto *Ap = &A[0][0];
  typedef real ArrTy[6][6];
  ArrTy *Aa = (ArrTy *)(Ap);

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



static void kernel(pbsize_t n, multarray<real, 2> A, multarray<real, 2> X) {
  //  auto *Ap =& A[0][0];
  // typedef real ArrTy[6][6];
  // ArrTy *Aa = (ArrTy*)( Ap);

  auto suggest_blocksize = [n](idx_t i) -> pbsize_t {
    return std::max(n / omp_get_num_threads(), 2);
  };



#pragma omp parallel default(none) shared(suggest_blocksize) firstprivate(n, A, X)
  {
    for (idx_t ii = 0; ii < n;) {
      auto blocksize = suggest_blocksize(ii);
      idx_t ni = std::min<idx_t>(blocksize, n - ii);



      // Step 1.
#pragma omp single
      {
        // printf("Inside task ii=%d tid=%d\n", ii, omp_get_thread_num());
#pragma omp task depend(inout: *&A[ii][ii]) default(none) firstprivate(ii, ni, A)
        cholesky_base(ii, ni, A); // A[ii..ii+n][ii..ii+n] <- cholesky(A[ii..ii+n][ii..ii+n])



        // Step 2.
        for (idx_t jj = ii + ni; jj < n; jj += blocksize) {
          idx_t nj = std::min<idx_t>(blocksize, n - jj);


          // A[jj..jj+n][ii..ii+n] <- A[jj..jj+n][ii..ii+n] * A[ii..ii+n][ii..ii+n]^-1
          //          B            <-        B                        A^-1
          //        L21_i          <-       A21_i           *        L11_T^1
#pragma omp task depend(inout:*&A[jj][ii]) depend(in:*&A[ii][ii]) default(none) firstprivate(ii, jj, nj, ni, A, X)
          my_dtrsm(ii, ii,
                   jj, ii,
                   nj, ni, A, X);
        }



        // Step 3. Case jj == kk
        for (idx_t jj = ii + ni; jj < n; jj += blocksize) {
          idx_t nj = std::min<idx_t>(blocksize, n - jj);

#pragma omp task depend(inout: *&A[jj][ii]) depend(in: *&A[ii][ii]) default(none) firstprivate(ii, jj, nj, ni, A)
          cholesky_syrk(
              jj, ii,
              jj, ii,
              jj, jj,
              nj, ni, A);

          //    }


          // Step 3. Case jj != kk
          // for (int jj = ii + ni; jj < n; jj += blocksize) {
          //   auto nj = std::min(blocksize, n - jj);
          for (idx_t kk = jj + blocksize; kk < n; kk += blocksize) {
            idx_t nk = std::min<idx_t>(blocksize, n - kk);

            // A[kk..kk+nk][jj..jj+nj] <- A[kk..kk+nk][jj..jj+nj] - A[jj..jj+nj][ii..ii+ni] * A[kk..kk+nj][ii..ii+ni]^T (UR space)
            // A[jj..jj+nj][kk..kk+nk] <- A[jj..jj+nj][kk..kk+nk] - A[ii..ii+ni][jj..jj+nj] * A[ii..ii+ni][kk..kk+nj]^T (LL space)
            //          C              <-         C               -   A   *   B
            //       A22_j,k           <-      A22_j,k            - L21_j * L21_k^T
            //       A22_i,j           <-      A22_i,j            - L21_i * L21_j^T
#pragma omp task default(none) firstprivate(ii, jj, kk, nk, nj, ni, A) \
                 depend(inout: *&A[kk][jj])                            \
                 depend(in: *&A[kk][ii])                               \
                 depend(in: *&A[jj][ii])
            cholesky_gemm(
                kk, ii,
                jj, ii,
                kk, jj,
                nk, nj, ni, A);
          }
        }
      }

      ii += blocksize;
    }
  }
}



void run(State &state, int pbsize) {
  pbsize_t n = pbsize; // 2000

  auto A = state.allocate_array<real>({n, n}, /*fakedata*/ true, /*verify*/ true, "A");
  auto tmp = state.allocate_array<real>({n, n}, /*fakedata*/ false, /*verify*/ false, "tmp"); // TODO: blocksize/allocate locally


  for (auto &&_ : state.manual()) {
    ensure_posdefinite(n, A);
    {
      auto &&scope = _.scope();
      // FIXME: cholesky of pos-definite matrix is not necessarily itself pos-definite
      kernel(n, A, tmp);
    }
  }
}
