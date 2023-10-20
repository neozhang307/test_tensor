#include <assert.h>
#include <iostream>
#include "stdio.h"
// #include "launchHelper.cuh"
// // CBLAS (OpenBLAS)
// #include "cblas.h"

// CUDA runtime
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// CUBLAS
#include <cublas_v2.h>

// NVML
#include <nvml.h>

/**
 * Panic wrapper for unwinding CUDA runtime errors
 */
#define CUDA_CHECK(status)                                                  \
    {                                                                       \
        cudaError_t error = status;                                         \
        if (error != cudaSuccess) {                                         \
            printf ("ERROR : %s %d CUDA : %s\n", __FILE__,  __LINE__, cudaGetErrorString(error));           \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    }

/**
 * Panic wrapper for unwinding CUBLAS runtime errors
 */
#define CUBLAS_CHECK(status)                                        \
    {                                                     \
        cublasStatus_t error = status;                                  \
        if(error != CUBLAS_STATUS_SUCCESS) {                        \
            printf ("ERROR : %s %d CUBLAS", __FILE__,  __LINE__);       \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    }


typedef double typ;

void init(typ *buf, int size) {
    for (int i = 0; i < size; ++i) {
        buf[i] = (typ)1.0f * rand() / RAND_MAX;
        //buf[i] = 1.0f;
    }
}

static float peak_flops;
static float get_peak_flops(){
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

  int Clock = prop.clockRate;
  int SM_nums = prop.multiProcessorCount;
    // int FP_units_pre_SM = 4 * 16 * 2; // RTX 3090 w/o tensor core
    int FP_units_pre_SM = 4 * 16; // A100 w/o tensor core
  float gflops = 2.0 * Clock * SM_nums * FP_units_pre_SM / 1e6;

    printf("Name:\t%s\n", prop.name);
    printf("Clock rate:\t%d\n", Clock);
    printf("Multiprocessor count:\t%d\n", SM_nums);
    printf("FP units count pre SM:\t%d\n", FP_units_pre_SM);
  printf("peakGFLOPs: %f\n", gflops);
  return gflops;
}

void becnmark_cublas(int M, int N, int K, int n_loops) {
    const int lda = K;
    const int ldb = N;
    const int ldc = N;
    const int a_alloc = sizeof(typ) * M * lda;
    const int b_alloc = sizeof(typ) * K * ldb;
    const int c_alloc = sizeof(typ) * M * ldc;

    typ* h_A =    (typ*)malloc(a_alloc);
    typ* h_B =    (typ*)malloc(b_alloc);
    typ* h_C =    (typ*)malloc(c_alloc);
    typ* h_refC = (typ*)malloc(c_alloc);
    typ alpha = 1.0;
    typ beta = 0.0;

    init(h_A, M * lda);
    init(h_B, K * ldb);

    typ* d_A;
    typ* d_B;
    typ* d_C;

    CUDA_CHECK(cudaMalloc(&d_A, a_alloc));
    CUDA_CHECK(cudaMalloc(&d_B, b_alloc));
    CUDA_CHECK(cudaMalloc(&d_C, c_alloc));
    
    CUDA_CHECK(cudaMemcpy( d_A, h_A, a_alloc, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy( d_B, h_B, b_alloc, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset( d_C, 0.f, c_alloc));

    // Initialization power
    unsigned int power1, power2;
    nvmlReturn_t result;
  nvmlDevice_t device;
  nvmlEnableState_t mode;

    result=nvmlInit();
  result = nvmlDeviceGetHandleByIndex(0, &device);
  assert(NVML_SUCCESS == result);
  result=nvmlDeviceGetPowerManagementMode(device, &mode);
  // printf("enabled = %d\n", mode);
  result=nvmlDeviceGetPowerUsage(device,&power1);
  assert(NVML_SUCCESS == result);
  cudaDeviceSynchronize();

    // Initialization timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    float msecTotal = 0;

    // cublas
    cublasHandle_t blas_handle;  
    CUBLAS_CHECK(cublasCreate(&blas_handle));
    // CUBLAS_CHECK(cublasSetMathMode( blas_handle, CUBLAS_TENSOR_OP_MATH ));
    // CUDA_CHECK(cudaMemcpy( d_C, h_refC, c_alloc, cudaMemcpyHostToDevice));
    for (int run = 0 ; run < n_loops; run ++ ) {
        CUBLAS_CHECK(
            cublasDgemm (blas_handle, CUBLAS_OP_N, CUBLAS_OP_N, 
                N, M, K, &alpha, 
                d_B, ldb, d_A, lda, &beta, d_C, ldc
            )
        );
    }
    
    CUDA_CHECK(cudaEventRecord(start));
    for (int run = 0 ; run < n_loops; run ++ ) {
        CUBLAS_CHECK(
            cublasDgemm (blas_handle, CUBLAS_OP_N, CUBLAS_OP_N, 
                N, M, K, &alpha, 
                d_B, ldb, d_A, lda, &beta, d_C, ldc
            )
        );
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&msecTotal, start, stop));

    cudaDeviceSynchronize();
  result=nvmlDeviceGetPowerUsage(device,&power2);
  assert(NVML_SUCCESS == result);
  nvmlShutdown();

    CUDA_CHECK(cudaMemcpy( h_C, d_C, c_alloc, cudaMemcpyDeviceToHost));
    CUBLAS_CHECK(cublasDestroy(blas_handle)); 

    float latency = msecTotal;
    float tflops = 2.0 * M * N * K / latency / 1e6 * n_loops/1000;
    printf("CUBLAS, M: %d, N: %d, K: %d, perf: %.2f tflops,  latency: %.6f ms, power from %u W to %u W\n", 
                      M, N, K,                  tflops,  latency / n_loops, power1/1000, power2/1000);
    // printf("%f, ", gflops);
    
    // // cblas
    // cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, alpha, h_A, lda, h_B, ldb, beta, h_refC, ldc);

    // double eps = 1.e-6;  // machine zero
    // bool correct = true;
    // for (int i = 0; i < M * N; i++) {
    //     double abs_err = fabs(h_C[i] - h_refC[i]);
    //     double dot_length = M;
    //     double abs_val = fabs(h_C[i]);
    //     double rel_err = abs_err / abs_val / dot_length;
    //     if (rel_err > eps) {
    //         printf("Error! Matrix[%05d]=%.8f, ref=%.8f error term is > %E\n",
    //                 i, h_C[i], h_refC[i], eps);
    //         correct = false;
    //         break;
    //     }
    // }
    // printf("%s\n", correct ? "Result= PASS" : "Result= FAIL");

    // Free Memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_refC);
}

int main(void) {

    int m = 10240;
    int n = 10240;
    int k = 2048;
    int repeats = 100;

    peak_flops = get_peak_flops();
    becnmark_cublas(m, n, k, repeats);

  return 0;
}
