#include "npy_cblas.h"
#include <stdio.h>
#include <string.h>
#include <sys/time.h>
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "symbols.h"
#include "wrapper.h"

const char* sym_name = "cblas_dgemm";

typedef void (*cblas_dgemm_f)(const enum CBLAS_ORDER, const enum CBLAS_TRANSPOSE,
			      const enum CBLAS_TRANSPOSE, const int, const int,
			      const int, const double, const double*,
			      const int, const double*, const int,
			      const double, double*, const int);

typedef void (*cblas_sgemm_f)(const enum CBLAS_ORDER, const enum CBLAS_TRANSPOSE,
			      const enum CBLAS_TRANSPOSE, const int, const int,
			      const int, const float, const float*,
			      const int, const float*, const int,
			      const float, float*, const int);

cblas_dgemm_f cblas_dgemm_sym = NULL;
cblas_sgemm_f cblas_sgemm_sym = NULL;


static void dmatmul(const double* a, const double* b, double* c,
		    size_t m, size_t n, size_t k)
{
  double* g_a;
  cudaMalloc((void**) &g_a, m * k * sizeof(double));
  double* g_b;
  cudaMalloc((void**) &g_b, k * n * sizeof(double));
  double* g_c;
  cudaMalloc((void**) &g_c, m * n * sizeof(double));

  cudaMemcpy(g_a, a, m * k * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(g_b, b, k * n * sizeof(double), cudaMemcpyHostToDevice);

  cublasHandle_t handle;
  cublasCreate(&handle);

  double alpha = 1;
  double beta = 0;

  cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k,
	      &alpha, g_b, n, g_a, k, &beta, g_c, n);

  cudaMemcpy(c, g_c, m * n * sizeof(double), cudaMemcpyDeviceToHost);
  cudaFree(g_a);
  cudaFree(g_b);
  cudaFree(g_c);
  
  cublasDestroy(handle);
}

static void smatmul(const float* a, const float* b, float* c,
		    size_t m, size_t n, size_t k)
{
  float* g_a;
  cudaMalloc((void**) &g_a, m * k * sizeof(float));
  float* g_b;
  cudaMalloc((void**) &g_b, k * n * sizeof(float));
  float* g_c;
  cudaMalloc((void**) &g_c, m * n * sizeof(float));

  cudaMemcpy(g_a, a, m * k * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(g_b, b, k * n * sizeof(float), cudaMemcpyHostToDevice);

  cublasHandle_t handle;
  cublasCreate(&handle);

  float alpha = 1;
  float beta = 0;

  cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k,
	      &alpha, g_b, n, g_a, k, &beta, g_c, n);

  cudaMemcpy(c, g_c, m * n * sizeof(float), cudaMemcpyDeviceToHost);
  cudaFree(g_a);
  cudaFree(g_b);
  cudaFree(g_c);
  
  cublasDestroy(handle);
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
  dmatmul(A, B, C, M, N, K);
}

void cblas_sgemm(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA,
                 const enum CBLAS_TRANSPOSE TransB, const int M, const int N,
                 const int K, const float alpha, const float *A,
                 const int lda, const float *B, const int ldb,
                 const float beta, float *C, const int ldc)
{

  if (cblas_sgemm_sym == NULL)
  {
    void* sym = find_symbol("cblas_sgemm");
    if (!sym)
    {
      printf("Failed to load symbol %s\n", "cblas_sgemm");
    }
    cblas_sgemm_sym = sym;
  }

  if (!g_use_wrapper)
  {
    cblas_sgemm_sym(Order, TransA, TransB, M, N, K, alpha, A, lda,B, ldb, beta, C, ldc);
    return;
  }
  
  printf("wrapper_cblas_sgemm()\n");
  smatmul(A, B, C, M, N, K);
}
