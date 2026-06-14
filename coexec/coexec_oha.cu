/*
 * One-head attention: out = (softmax(Q * K^T)) * V.
 * Sizes (HeteroBench): BATCH=8, N=1024, D=128.
 * The two batched matmuls (Q*K^T and S*V) are co-executed by splitting the
 * flattened output-row domain (BATCH*N = 8192) between GPU [0,split) and CPU.
 * The row-wise softmax (cheap) runs on the host with OpenMP.
 *
 * Usage: ./coexec_oha <frac> <reps>  -> CSV: oha,frac,t_total,t_cpu,t_gpu,checksum
 */
#include "coexec_common.h"

static const int BATCH = 8, N = 1024, Dh = 128;
static const int M = BATCH * N;  // 8192 output rows

// O[r][k] = sum_{l<D} Q[r][l] * key[b][k][l]   (folds the K transpose in)
__global__ void bmm1_kernel(const float *Q, const float *key, float *O,
                            int rows, int Nn, int D) {
  int k = blockIdx.x * blockDim.x + threadIdx.x;  // 0..N-1
  int r = blockIdx.y * blockDim.y + threadIdx.y;  // 0..rows-1
  if (r >= rows || k >= Nn) return;
  int b = r / Nn;
  const float *Qr = Q + (size_t)r * D;
  const float *Kk = key + ((size_t)b * Nn + k) * D;
  float acc = 0.0f;
  for (int l = 0; l < D; ++l) acc += Qr[l] * Kk[l];
  O[(size_t)r * Nn + k] = acc;
}

// out[r][k] = sum_{l<N} S[r][l] * V[b][l][k]
__global__ void bmm2_kernel(const float *S, const float *V, float *out,
                            int rows, int Nn, int D) {
  int k = blockIdx.x * blockDim.x + threadIdx.x;  // 0..D-1
  int r = blockIdx.y * blockDim.y + threadIdx.y;  // 0..rows-1
  if (r >= rows || k >= D) return;
  int b = r / Nn;
  const float *Sr = S + (size_t)r * Nn;
  float acc = 0.0f;
  for (int l = 0; l < Nn; ++l) acc += Sr[l] * V[((size_t)b * Nn + l) * D + k];
  out[(size_t)r * D + k] = acc;
}

static void cpu_bmm1(const float *Q, const float *key, float *O, int r0, int r1) {
#pragma omp parallel for schedule(static)
  for (int r = r0; r < r1; ++r) {
    int b = r / N;
    const float *Qr = Q + (size_t)r * Dh;
    for (int k = 0; k < N; ++k) {
      const float *Kk = key + ((size_t)b * N + k) * Dh;
      float acc = 0.0f;
      for (int l = 0; l < Dh; ++l) acc += Qr[l] * Kk[l];
      O[(size_t)r * N + k] = acc;
    }
  }
}

static void cpu_bmm2(const float *S, const float *V, float *out, int r0, int r1) {
#pragma omp parallel for schedule(static)
  for (int r = r0; r < r1; ++r) {
    int b = r / N;
    const float *Sr = S + (size_t)r * N;
    for (int k = 0; k < Dh; ++k) {
      float acc = 0.0f;
      for (int l = 0; l < N; ++l) acc += Sr[l] * V[((size_t)b * N + l) * Dh + k];
      out[(size_t)r * Dh + k] = acc;
    }
  }
}

static void softmax_rows(float *X, int rows_total) {
#pragma omp parallel for schedule(static)
  for (int r = 0; r < rows_total; ++r) {
    float *Xr = X + (size_t)r * N;
    float m = Xr[0];
    for (int j = 1; j < N; ++j) m = fmaxf(m, Xr[j]);
    float sum = 0.0f;
    for (int j = 0; j < N; ++j) { Xr[j] = expf(Xr[j] - m); sum += Xr[j]; }
    float inv = 1.0f / sum;
    for (int j = 0; j < N; ++j) Xr[j] *= inv;
  }
}

int main(int argc, char **argv) {
  double f = (argc > 1) ? atof(argv[1]) : 0.5;
  int reps = (argc > 2) ? atoi(argv[2]) : 3;

  float *Q, *K, *V, *S, *Out;
  size_t qb = (size_t)BATCH * N * Dh, sb = (size_t)BATCH * N * N;
  CUDA_CHECK(cudaMallocHost(&Q, qb * sizeof(float)));
  CUDA_CHECK(cudaMallocHost(&K, qb * sizeof(float)));
  CUDA_CHECK(cudaMallocHost(&V, qb * sizeof(float)));
  CUDA_CHECK(cudaMallocHost(&S, sb * sizeof(float)));
  CUDA_CHECK(cudaMallocHost(&Out, qb * sizeof(float)));
  for (size_t i = 0; i < qb; ++i) {
    Q[i] = (float)((i % 13) - 6) * 0.05f; K[i] = (float)((i % 7) - 3) * 0.05f;
    V[i] = (float)((i % 11) - 5) * 0.05f;
  }

  float *dQ, *dK, *dV, *dS, *dOut;
  CUDA_CHECK(cudaMalloc(&dQ, qb * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dK, qb * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dV, qb * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dS, sb * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dOut, qb * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(dK, K, qb * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dV, V, qb * sizeof(float), cudaMemcpyHostToDevice));
  cudaStream_t s; CUDA_CHECK(cudaStreamCreate(&s));
  int split = gpu_split(f, M);

  std::vector<double> tot, tcv, tgv;
  for (int r = 0; r < reps + 1; ++r) {
    double tc = 0, tg = 0, c1, g1;
    auto t0 = clk::now();
    // Stage 1: S = Q * K^T  (split rows)
    auto gpu1 = [&]() {
      if (split <= 0) return;
      CUDA_CHECK(cudaMemcpyAsync(dQ, Q, (size_t)split * Dh * sizeof(float),
                                 cudaMemcpyHostToDevice, s));
      dim3 b(16, 16), g((N + 15) / 16, (split + 15) / 16);
      bmm1_kernel<<<g, b, 0, s>>>(dQ, dK, dS, split, N, Dh);
      CUDA_CHECK(cudaMemcpyAsync(S, dS, (size_t)split * N * sizeof(float),
                                 cudaMemcpyDeviceToHost, s));
      CUDA_CHECK(cudaStreamSynchronize(s));
    };
    auto cpu1 = [&]() { if (split < M) cpu_bmm1(Q, K, S, split, M); };
    run_concurrent(split < M, split > 0, cpu1, gpu1, c1, g1); tc += c1; tg += g1;

    // Stage 2: row-wise softmax of S (host)
    auto ts0 = clk::now();
    softmax_rows(S, M);
    tc += ms_since(ts0);

    // Stage 3: Out = S * V  (split rows); push full S to device first.
    auto gpu2 = [&]() {
      if (split <= 0) return;
      CUDA_CHECK(cudaMemcpyAsync(dS, S, (size_t)split * N * sizeof(float),
                                 cudaMemcpyHostToDevice, s));
      dim3 b(16, 16), g((Dh + 15) / 16, (split + 15) / 16);
      bmm2_kernel<<<g, b, 0, s>>>(dS, dV, dOut, split, N, Dh);
      CUDA_CHECK(cudaMemcpyAsync(Out, dOut, (size_t)split * Dh * sizeof(float),
                                 cudaMemcpyDeviceToHost, s));
      CUDA_CHECK(cudaStreamSynchronize(s));
    };
    auto cpu2 = [&]() { if (split < M) cpu_bmm2(S, V, Out, split, M); };
    run_concurrent(split < M, split > 0, cpu2, gpu2, c1, g1); tc += c1; tg += g1;

    double tt = ms_since(t0);
    if (r > 0) { tot.push_back(tt); tcv.push_back(tc); tgv.push_back(tg); }
  }
  double chk = 0; for (int i = 0; i < M; i += 137) chk += Out[(size_t)i * Dh + (i % Dh)];
  Result res{median(tot), median(tcv), median(tgv), chk};
  emit_csv("oha", f, res);
  return 0;
}
