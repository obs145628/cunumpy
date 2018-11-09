#include <stdio.h>
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>

void print_fmat(const float* m, size_t rows, size_t cols)
{
  for (size_t i = 0; i < rows; ++i)
  {
    printf("|");
    for (size_t j = 0; j < cols; ++j)
      printf("%G |", m[i * cols + j]);
    printf("\n");
  }
}

void fmatmul(const float* a, const float* b, float* c,
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

  float elapsed = 0;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);

  int status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k,
			   &alpha, g_b, n, g_a, k, &beta, g_c, n);

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsed, start, stop);

  cudaMemcpy(c, g_c, m * n * sizeof(float), cudaMemcpyDeviceToHost);
  cudaFree(g_a);
  cudaFree(g_b);
  cudaFree(g_c);
  
  printf("status = %d, duration = %G ms\n", status, (double) elapsed);
  
  cublasDestroy(handle);
}

void print_dmat(const double* m, size_t rows, size_t cols)
{
  for (size_t i = 0; i < rows; ++i)
  {
    printf("|");
    for (size_t j = 0; j < cols; ++j)
      printf("%G |", m[i * cols + j]);
    printf("\n");
  }
}

void dmatmul(const double* a, const double* b, double* c,
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

  float elapsed = 0;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);

  int status = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k,
			   &alpha, g_b, n, g_a, k, &beta, g_c, n);

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsed, start, stop);

  cudaMemcpy(c, g_c, m * n * sizeof(double), cudaMemcpyDeviceToHost);
  cudaFree(g_a);
  cudaFree(g_b);
  cudaFree(g_c);
  
  printf("status = %d, duration = %G ms\n", status, (double) elapsed);
  
  cublasDestroy(handle);
}

void drand(double* begin, double* end)
{
  for (double* it = begin; it != end; ++it)
    *it = (double) (rand() % 100) / 100;
}

void frand(float* begin, float* end)
{
  for (float* it = begin; it != end; ++it)
    *it = (float) (rand() % 100) / 100;
}

int main()
{

  size_t a_rows = 8000;
  size_t a_cols = 8000;
  size_t b_rows = a_cols;
  size_t b_cols = 8000;

  float* mat_a = malloc(a_rows * a_cols * sizeof(float));
  float* mat_b = malloc(b_rows * b_cols * sizeof(float));
  float* mat_c = malloc(a_rows * b_cols * sizeof(float));
  frand(mat_a, mat_a + a_rows * a_cols);
  frand(mat_b, mat_b + b_rows * b_cols);

  fmatmul(mat_a, mat_b, mat_c, a_rows, b_cols, a_cols);

  /*
  printf("A=\n");
  print_dmat(mat_a, a_rows, a_cols);
  printf("\n");

  printf("B=\n");
  print_dmat(mat_b, b_rows, b_cols);
  printf("\n");

  printf("C=\n");
  print_dmat(mat_c, a_rows, b_cols);
  printf("\n");
  */

  free(mat_a);
  free(mat_b);
  free(mat_c);
}
