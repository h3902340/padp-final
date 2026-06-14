/*
 * MLP forward pass: 4 fully-connected layers (dot_add = matmul + bias), with
 * sigmoid activations on layers 0-2 and a softmax on layer 3.
 * Sizes (HeteroBench): batch H1=3072; widths 2048->4096->4096->4096->1024.
 * Each layer's matmul splits output ROWS (the batch dim) between GPU and CPU;
 * the cheap activations run on the host after each stage.
 *
 * Usage: ./coexec_mlp <frac> <reps>  -> CSV: mlp,frac,t_total,t_cpu,t_gpu,checksum
 */
#include "coexec_common.h"

#define TILE 16
static const int H = 3072;
static const int Ws[5] = {2048, 4096, 4096, 4096, 1024};
static const float SCALE[4] = {500.f, 1500.f, 1500.f, 1500.f};

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

int main(int argc, char **argv) {
  double f = (argc > 1) ? atof(argv[1]) : 0.5;
  int reps = (argc > 2) ? atoi(argv[2]) : 1;

  int maxW = 4096;
  float *x, *a, *z;  // activations (reused per layer, max size H*maxW)
  CUDA_CHECK(cudaMallocHost(&x, (size_t)H * maxW * sizeof(float)));
  CUDA_CHECK(cudaMallocHost(&a, (size_t)H * maxW * sizeof(float)));
  CUDA_CHECK(cudaMallocHost(&z, (size_t)H * maxW * sizeof(float)));
  float *W[4], *b[4], *dW[4];
  for (int l = 0; l < 4; ++l) {
    CUDA_CHECK(cudaMallocHost(&W[l], (size_t)Ws[l] * Ws[l + 1] * sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&b[l], (size_t)Ws[l + 1] * sizeof(float)));
    for (size_t i = 0; i < (size_t)Ws[l] * Ws[l + 1]; ++i)
      W[l][i] = (float)((i % 17) - 8) * 0.002f;
    for (int j = 0; j < Ws[l + 1]; ++j) b[l][j] = (float)((j % 9) - 4) * 0.01f;
    CUDA_CHECK(cudaMalloc(&dW[l], (size_t)Ws[l] * Ws[l + 1] * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dW[l], W[l], (size_t)Ws[l] * Ws[l + 1] * sizeof(float),
                          cudaMemcpyHostToDevice));
  }
  for (size_t i = 0; i < (size_t)H * Ws[0]; ++i) x[i] = (float)((i % 13) - 6) * 0.02f;

  float *dIn, *dOut;
  CUDA_CHECK(cudaMalloc(&dIn, (size_t)H * maxW * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dOut, (size_t)H * maxW * sizeof(float)));
  cudaStream_t s; CUDA_CHECK(cudaStreamCreate(&s));
  int split = gpu_split(f, H);

  std::vector<double> tot, tcv, tgv;
  for (int r = 0; r < reps + 1; ++r) {
    double tc = 0, tg = 0;
    auto t0 = clk::now();
    float *in = x;  // layer input (host); a = raw matmul out; z = activated
    for (int l = 0; l < 4; ++l) {
      int K = Ws[l], N = Ws[l + 1];
      double c1, g1;
      auto gpu = [&]() {
        if (split <= 0) return;
        CUDA_CHECK(cudaMemcpyAsync(dIn, in, (size_t)split * K * sizeof(float),
                                   cudaMemcpyHostToDevice, s));
        dim3 bl(TILE, TILE), gr((N + TILE - 1) / TILE, (split + TILE - 1) / TILE);
        mm_kernel<<<gr, bl, 0, s>>>(dIn, dW[l], dOut, split, K, N);
        CUDA_CHECK(cudaMemcpyAsync(a, dOut, (size_t)split * N * sizeof(float),
                                   cudaMemcpyDeviceToHost, s));
        CUDA_CHECK(cudaStreamSynchronize(s));
      };
      auto cpu = [&]() { if (split < H) cpu_mm(in, W[l], a, split, H, K, N); };
      run_concurrent(split < H, split > 0, cpu, gpu, c1, g1);
      tc += c1; tg += g1;

      // Activation (host, OpenMP): add bias, scale, then sigmoid/softmax.
      if (l < 3) {
#pragma omp parallel for schedule(static)
        for (int i = 0; i < H; ++i)
          for (int j = 0; j < N; ++j) {
            float v = (a[(size_t)i * N + j] + b[l][j]) / SCALE[l];
            z[(size_t)i * N + j] = 1.0f / (1.0f + expf(-v));
          }
      } else {
#pragma omp parallel for schedule(static)
        for (int i = 0; i < H; ++i) {
          float *ai = a + (size_t)i * N;
          double sum = 0;
          for (int j = 0; j < N; ++j) {
            ai[j] = (ai[j] + b[l][j]) / SCALE[l];
            sum += exp((double)ai[j]);
          }
          for (int j = 0; j < N; ++j)
            z[(size_t)i * N + j] = (float)(exp((double)ai[j]) / sum * 1e6);
        }
      }
      in = z;
    }
    double tt = ms_since(t0);
    if (r > 0) { tot.push_back(tt); tcv.push_back(tc); tgv.push_back(tg); }
  }
  double chk = 0; for (int i = 0; i < H; i += 131) chk += z[(size_t)i * Ws[4] + (i % Ws[4])];
  Result res{median(tot), median(tcv), median(tgv), chk};
  emit_csv("mlp", f, res);
  return 0;
}
