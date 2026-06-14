/*
 * Concurrent CPU+GPU co-execution of single-precision GEMM (C = A * B).
 *
 * The output rows of C are partitioned by a fraction f:
 *   - GPU computes rows [0, split)         (split = round(f * N))
 *   - CPU computes rows [split, N)          on all OpenMP threads
 * Both run AT THE SAME TIME: a host std::thread enqueues the GPU work and
 * blocks on the GPU, while the main thread drives the CPU OpenMP loop.
 *
 * This is a COMPUTE-BOUND kernel (arithmetic intensity ~ N/2 FLOP/byte),
 * representative of HeteroBench 3mm / mlp / cnn dense-matmul kernels.
 *
 * Usage: ./coexec_gemm N frac reps
 *   N    : matrix dimension (square)
 *   frac : fraction of rows assigned to the GPU in [0,1]
 *   reps : timed repetitions (median reported)
 *
 * Output (one CSV line): gemm,N,frac,t_total_ms,t_cpu_ms,t_gpu_ms,gflops
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#include <thread>
#include <chrono>
#include <cuda_runtime.h>
#include <omp.h>

#define TILE 16

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = (call);                                                  \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,            \
              cudaGetErrorString(err));                                        \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

__global__ void gemm_tiled(const float *A, const float *B, float *C,
                           int rows, int N) {
  __shared__ float As[TILE][TILE];
  __shared__ float Bs[TILE][TILE];
  int row = blockIdx.y * TILE + threadIdx.y;
  int col = blockIdx.x * TILE + threadIdx.x;
  float acc = 0.0f;
  for (int t = 0; t < (N + TILE - 1) / TILE; ++t) {
    int aCol = t * TILE + threadIdx.x;
    int bRow = t * TILE + threadIdx.y;
    As[threadIdx.y][threadIdx.x] =
        (row < rows && aCol < N) ? A[row * N + aCol] : 0.0f;
    Bs[threadIdx.y][threadIdx.x] =
        (bRow < N && col < N) ? B[bRow * N + col] : 0.0f;
    __syncthreads();
#pragma unroll
    for (int k = 0; k < TILE; ++k) acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
    __syncthreads();
  }
  if (row < rows && col < N) C[row * N + col] = acc;
}

static void cpu_gemm(const float *A, const float *B, float *C,
                     int row0, int row1, int N) {
#pragma omp parallel for schedule(static)
  for (int i = row0; i < row1; ++i) {
    for (int j = 0; j < N; ++j) C[i * N + j] = 0.0f;
    for (int k = 0; k < N; ++k) {
      float a = A[i * N + k];
      const float *Brow = B + k * N;
      float *Crow = C + i * N;
      for (int j = 0; j < N; ++j) Crow[j] += a * Brow[j];
    }
  }
}

int main(int argc, char **argv) {
  int N = (argc > 1) ? atoi(argv[1]) : 1024;
  double frac = (argc > 2) ? atof(argv[2]) : 0.5;
  int reps = (argc > 3) ? atoi(argv[3]) : 3;
  if (frac < 0) frac = 0;
  if (frac > 1) frac = 1;
  int split = (int)llround(frac * N);  // rows on GPU

  size_t bytesFull = (size_t)N * N * sizeof(float);
  float *A, *B, *C;
  CUDA_CHECK(cudaMallocHost(&A, bytesFull));
  CUDA_CHECK(cudaMallocHost(&B, bytesFull));
  CUDA_CHECK(cudaMallocHost(&C, bytesFull));
  for (size_t i = 0; i < (size_t)N * N; ++i) {
    A[i] = (float)((i % 13) - 6) * 0.1f;
    B[i] = (float)((i % 7) - 3) * 0.1f;
  }

  // GPU buffers (only need 'split' rows of A and C, full B)
  float *dA = nullptr, *dB = nullptr, *dC = nullptr;
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));
  if (split > 0) {
    CUDA_CHECK(cudaMalloc(&dA, (size_t)split * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dC, (size_t)split * N * sizeof(float)));
  }
  CUDA_CHECK(cudaMalloc(&dB, bytesFull));
  CUDA_CHECK(cudaMemcpy(dB, B, bytesFull, cudaMemcpyHostToDevice));

  std::vector<double> totals, cputimes, gputimes;
  for (int r = 0; r < reps + 1; ++r) {  // +1 warmup
    double t_cpu = 0, t_gpu = 0;
    auto t0 = std::chrono::high_resolution_clock::now();

    // Launch GPU portion on a dedicated host thread (concurrent with CPU)
    std::thread gpu_thread;
    if (split > 0) {
      gpu_thread = std::thread([&]() {
        auto g0 = std::chrono::high_resolution_clock::now();
        CUDA_CHECK(cudaMemcpyAsync(dA, A, (size_t)split * N * sizeof(float),
                                   cudaMemcpyHostToDevice, stream));
        dim3 block(TILE, TILE);
        dim3 grid((N + TILE - 1) / TILE, (split + TILE - 1) / TILE);
        gemm_tiled<<<grid, block, 0, stream>>>(dA, dB, dC, split, N);
        CUDA_CHECK(cudaMemcpyAsync(C, dC, (size_t)split * N * sizeof(float),
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        auto g1 = std::chrono::high_resolution_clock::now();
        t_gpu = std::chrono::duration<double, std::milli>(g1 - g0).count();
      });
    }

    // CPU portion concurrently on main thread (OpenMP)
    auto c0 = std::chrono::high_resolution_clock::now();
    if (split < N) cpu_gemm(A, B, C, split, N, N);
    auto c1 = std::chrono::high_resolution_clock::now();
    t_cpu = std::chrono::duration<double, std::milli>(c1 - c0).count();

    if (gpu_thread.joinable()) gpu_thread.join();
    auto t1 = std::chrono::high_resolution_clock::now();
    double t_total = std::chrono::duration<double, std::milli>(t1 - t0).count();

    if (r > 0) {  // skip warmup
      totals.push_back(t_total);
      cputimes.push_back(t_cpu);
      gputimes.push_back(t_gpu);
    }
  }

  auto median = [](std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v.empty() ? 0.0 : v[v.size() / 2];
  };
  double t_total = median(totals);
  double t_cpu = median(cputimes);
  double t_gpu = median(gputimes);
  double gflops = (2.0 * N * N * N) / (t_total / 1000.0) / 1e9;

  printf("gemm,%d,%.4f,%.4f,%.4f,%.4f,%.2f\n", N, frac, t_total, t_cpu, t_gpu,
         gflops);

  if (dA) cudaFree(dA);
  if (dC) cudaFree(dC);
  cudaFree(dB);
  cudaFreeHost(A);
  cudaFreeHost(B);
  cudaFreeHost(C);
  cudaStreamDestroy(stream);
  return 0;
}
