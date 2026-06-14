/*
 * 3mm (PolyBench 3 matrix multiply): G = (A*B) * (C*D), all 1024x1024.
 * Pipeline: E=A*B, F=C*D (independent), then G=E*F.
 * Each matmul's output rows are split between GPU [0,split) and CPU [split,M).
 *
 * Usage: ./coexec_3mm <frac> <reps>   ->  CSV: 3mm,frac,t_total,t_cpu,t_gpu,checksum
 */
#include "coexec_common.h"

#define TILE 16
static const int NI = 1024, NJ = 1024, NK = 1024, NL = 1024, NM = 1024;

__global__ void mm_kernel(const float *A, const float *B, float *C, int rows,
                          int K, int N) {
  __shared__ float As[TILE][TILE], Bs[TILE][TILE];
  int row = blockIdx.y * TILE + threadIdx.y;
  int col = blockIdx.x * TILE + threadIdx.x;
  float acc = 0.0f;
  for (int t = 0; t < (K + TILE - 1) / TILE; ++t) {
    int aCol = t * TILE + threadIdx.x, bRow = t * TILE + threadIdx.y;
    As[threadIdx.y][threadIdx.x] = (row < rows && aCol < K) ? A[row * K + aCol] : 0.0f;
    Bs[threadIdx.y][threadIdx.x] = (bRow < K && col < N) ? B[bRow * N + col] : 0.0f;
    __syncthreads();
#pragma unroll
    for (int k = 0; k < TILE; ++k) acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
    __syncthreads();
  }
  if (row < rows && col < N) C[row * N + col] = acc;
}

static void cpu_mm(const float *A, const float *B, float *C, int r0, int r1,
                   int K, int N) {
#pragma omp parallel for schedule(static)
  for (int i = r0; i < r1; ++i) {
    float *Ci = C + (size_t)i * N;
    for (int j = 0; j < N; ++j) Ci[j] = 0.0f;
    for (int k = 0; k < K; ++k) {
      float a = A[(size_t)i * K + k];
      const float *Bk = B + (size_t)k * N;
      for (int j = 0; j < N; ++j) Ci[j] += a * Bk[j];
    }
  }
}

// One co-executed matmul stage: C[MxN] = A[MxK] * dB[KxN] (dB persistent on dev).
static void mm_stage(const float *A, float *dA, const float *dummyB, float *dB,
                     float *C, float *dC, cudaStream_t s, int M, int K, int N,
                     int split, double &tc, double &tg) {
  auto gpu = [&]() {
    if (split <= 0) return;
    CUDA_CHECK(cudaMemcpyAsync(dA, A, (size_t)split * K * sizeof(float),
                               cudaMemcpyHostToDevice, s));
    dim3 b(TILE, TILE), g((N + TILE - 1) / TILE, (split + TILE - 1) / TILE);
    mm_kernel<<<g, b, 0, s>>>(dA, dB, dC, split, K, N);
    CUDA_CHECK(cudaMemcpyAsync(C, dC, (size_t)split * N * sizeof(float),
                               cudaMemcpyDeviceToHost, s));
    CUDA_CHECK(cudaStreamSynchronize(s));
  };
  auto cpu = [&]() {
    if (split < M) cpu_mm(A, dummyB, C, split, M, K, N);
  };
  run_concurrent(split < M, split > 0, cpu, gpu, tc, tg);
}

int main(int argc, char **argv) {
  double f = (argc > 1) ? atof(argv[1]) : 0.5;
  int reps = (argc > 2) ? atoi(argv[2]) : 3;

  float *A, *B, *C, *D, *E, *F, *G;
  size_t sz = (size_t)NI * NJ * sizeof(float);
  CUDA_CHECK(cudaMallocHost(&A, sz)); CUDA_CHECK(cudaMallocHost(&B, sz));
  CUDA_CHECK(cudaMallocHost(&C, sz)); CUDA_CHECK(cudaMallocHost(&D, sz));
  CUDA_CHECK(cudaMallocHost(&E, sz)); CUDA_CHECK(cudaMallocHost(&F, sz));
  CUDA_CHECK(cudaMallocHost(&G, sz));
  for (size_t i = 0; i < (size_t)NI * NJ; ++i) {
    A[i] = (float)((i % 13) - 6) * 0.01f; B[i] = (float)((i % 7) - 3) * 0.01f;
    C[i] = (float)((i % 11) - 5) * 0.01f; D[i] = (float)((i % 5) - 2) * 0.01f;
  }

  float *dA, *dB, *dCm, *dD, *dE, *dF, *dOut;
  cudaStream_t s; CUDA_CHECK(cudaStreamCreate(&s));
  CUDA_CHECK(cudaMalloc(&dA, sz)); CUDA_CHECK(cudaMalloc(&dB, sz));
  CUDA_CHECK(cudaMalloc(&dCm, sz)); CUDA_CHECK(cudaMalloc(&dD, sz));
  CUDA_CHECK(cudaMalloc(&dE, sz)); CUDA_CHECK(cudaMalloc(&dF, sz));
  CUDA_CHECK(cudaMalloc(&dOut, sz));
  // Persistent right operands on device.
  CUDA_CHECK(cudaMemcpy(dB, B, sz, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dD, D, sz, cudaMemcpyHostToDevice));

  int split = gpu_split(f, NI);
  std::vector<double> tot, tcv, tgv;
  for (int r = 0; r < reps + 1; ++r) {
    double tc = 0, tg = 0, a, b;
    auto t0 = clk::now();
    // E = A*B  and  F = C*D  (sequential split stages)
    mm_stage(A, dA, B, dB, E, dE, s, NI, NK, NJ, split, a, b); tc += a; tg += b;
    mm_stage(C, dCm, D, dD, F, dF, s, NJ, NM, NL, split, a, b); tc += a; tg += b;
    // G = E*F needs full F resident as the right operand: push CPU-computed rows.
    if (split < NJ)
      CUDA_CHECK(cudaMemcpy(dF + (size_t)split * NL, F + (size_t)split * NL,
                            (size_t)(NJ - split) * NL * sizeof(float),
                            cudaMemcpyHostToDevice));
    // G = E*F  (needs full E,F; dE/dF fully populated, host E/F fully populated)
    mm_stage(E, dA, F, dF, G, dOut, s, NI, NJ, NL, split, a, b); tc += a; tg += b;

    double tt = ms_since(t0);
    if (r > 0) { tot.push_back(tt); tcv.push_back(tc); tgv.push_back(tg); }
  }
  double chk = 0; for (int i = 0; i < NI; i += 97) chk += G[(size_t)i * NL + (i % NL)];
  Result res{median(tot), median(tcv), median(tgv), chk};
  emit_csv("3mm", f, res);

  cudaFree(dA); cudaFree(dB); cudaFree(dCm); cudaFree(dD); cudaFree(dE);
  cudaFree(dF); cudaFree(dOut); cudaStreamDestroy(s);
  cudaFreeHost(A); cudaFreeHost(B); cudaFreeHost(C); cudaFreeHost(D);
  cudaFreeHost(E); cudaFreeHost(F); cudaFreeHost(G);
  return 0;
}
