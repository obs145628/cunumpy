#include "npy_cblas.h"
#include <stdio.h>
#include <string.h>
#include <sys/time.h>
#include "symbols.h"
#include "wrapper.h"

const char* sym_name = "cblas_dgemm";

typedef void (*cblas_dgemm_f)(const enum CBLAS_ORDER, const enum CBLAS_TRANSPOSE,
			      const enum CBLAS_TRANSPOSE, const int, const int,
			      const int, const double, const double*,
			      const int, const double*, const int,
			      const double, double*, const int);

cblas_dgemm_f cblas_dgemm_sym = NULL;

static void my_matmul(const double* A, const double* B, double* C,
		      size_t M, size_t N, size_t K)
{
  for (size_t i = 0; i < M; ++i)
    for (size_t j = 0; j < N; ++j)
    {
      double val = 0;
      for (size_t k = 0; k < K; ++k)
	val += A[i * K + k] * B[k * N + j];
      C[i * N + j] = val;
    }
}

void cblas_dgemm(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA,
                 const enum CBLAS_TRANSPOSE TransB, const int M, const int N,
                 const int K, const double alpha, const double *A,
                 const int lda, const double *B, const int ldb,
                 const double beta, double *C, const int ldc)
{

  if (cblas_dgemm_sym == NULL)
  {
    void* sym = find_symbol(sym_name);
    if (!sym)
    {
      printf("Failed to load symbol %s\n", sym_name);
    }
    cblas_dgemm_sym = sym;
  }

  if (!g_use_wrapper)
  {
    cblas_dgemm_sym(Order, TransA, TransB, M, N, K, alpha, A, lda,B, ldb, beta, C, ldc);
    return;
  }
  
  printf("wrapper_cblas_dgemm()\n");
  //cblas_dgemm_sym(Order, TransA, TransB, M, N, K, alpha, A, lda,B, ldb, beta, C, ldc);
  my_matmul(A, B, C, M, N, K);
}
