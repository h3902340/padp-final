/*
 * Sobel filter: sobel_x (3x3) , sobel_y (3x3) , gradient magnitude = sqrt(sx^2+sy^2).
 * Size (HeteroBench stanford image): 1919 x 1439.
 * Each stage splits image ROWS between GPU [0,split) and CPU [split,H).
 *
 * Usage: ./coexec_sbf <frac> <reps>  -> CSV: sbf,frac,t_total,t_cpu,t_gpu,checksum
 */
#include "coexec_common.h"

static const int H = 1919, W = 1439;

__global__ void conv3x3_kernel(const float *in, const float *wt, float *out,
                               int rows, int Hh, int Ww) {
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= rows || j >= Ww) return;
  if (i == 0 || i == Hh - 1 || j == 0 || j == Ww - 1) { out[(size_t)i * Ww + j] = 0; return; }
  float acc = 0.0f;
  for (int ki = -1; ki <= 1; ++ki)
    for (int kj = -1; kj <= 1; ++kj)
      acc += in[(size_t)(i + ki) * Ww + (j + kj)] * wt[(ki + 1) * 3 + (kj + 1)];
  out[(size_t)i * Ww + j] = acc;
}

__global__ void mag_kernel(const float *sx, const float *sy, float *out, int rows, int Ww) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= rows * Ww) return;
  out[idx] = sqrtf(sx[idx] * sx[idx] + sy[idx] * sy[idx]);
}

static void cpu_conv3x3(const float *in, const float *wt, float *out, int r0,
                        int r1, int Hh, int Ww) {
#pragma omp parallel for schedule(static)
  for (int i = r0; i < r1; ++i)
    for (int j = 0; j < Ww; ++j) {
      if (i == 0 || i == Hh - 1 || j == 0 || j == Ww - 1) { out[(size_t)i * Ww + j] = 0; continue; }
      float acc = 0.0f;
      for (int ki = -1; ki <= 1; ++ki)
        for (int kj = -1; kj <= 1; ++kj)
          acc += in[(size_t)(i + ki) * Ww + (j + kj)] * wt[(ki + 1) * 3 + (kj + 1)];
      out[(size_t)i * Ww + j] = acc;
    }
}

static void cpu_mag(const float *sx, const float *sy, float *out, int e0, int e1) {
#pragma omp parallel for schedule(static)
  for (int i = e0; i < e1; ++i) out[i] = sqrtf(sx[i] * sx[i] + sy[i] * sy[i]);
}

int main(int argc, char **argv) {
  double f = (argc > 1) ? atof(argv[1]) : 0.5;
  int reps = (argc > 2) ? atoi(argv[2]) : 3;
  float kx[9] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
  float ky[9] = {-1, -2, -1, 0, 0, 0, 1, 2, 1};

  float *in, *sx, *sy, *mag;
  size_t sz = (size_t)H * W * sizeof(float);
  CUDA_CHECK(cudaMallocHost(&in, sz)); CUDA_CHECK(cudaMallocHost(&sx, sz));
  CUDA_CHECK(cudaMallocHost(&sy, sz)); CUDA_CHECK(cudaMallocHost(&mag, sz));
  for (size_t i = 0; i < (size_t)H * W; ++i) in[i] = (float)(i % 256);

  float *dIn, *dKx, *dKy, *dSx, *dSy, *dMag;
  CUDA_CHECK(cudaMalloc(&dIn, sz)); CUDA_CHECK(cudaMalloc(&dKx, 9 * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dKy, 9 * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dSx, sz)); CUDA_CHECK(cudaMalloc(&dSy, sz)); CUDA_CHECK(cudaMalloc(&dMag, sz));
  CUDA_CHECK(cudaMemcpy(dIn, in, sz, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dKx, kx, 9 * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dKy, ky, 9 * sizeof(float), cudaMemcpyHostToDevice));
  cudaStream_t s; CUDA_CHECK(cudaStreamCreate(&s));
  int split = gpu_split(f, H);
  int esplit = split * W;  // element split for magnitude

  std::vector<double> tot, tcv, tgv;
  for (int r = 0; r < reps + 1; ++r) {
    double tc = 0, tg = 0, c1, g1;
    auto t0 = clk::now();
    auto convStage = [&](const float *dW, float *dOut, float *hOut) {
      auto gpu = [&]() {
        if (split <= 0) return;
        dim3 b(16, 16), g((W + 15) / 16, (split + 15) / 16);
        conv3x3_kernel<<<g, b, 0, s>>>(dIn, dW, dOut, split, H, W);
        CUDA_CHECK(cudaMemcpyAsync(hOut, dOut, (size_t)split * W * sizeof(float),
                                   cudaMemcpyDeviceToHost, s));
        CUDA_CHECK(cudaStreamSynchronize(s));
      };
      const float *hW = (dW == dKx) ? kx : ky;
      auto cpu = [&]() { if (split < H) cpu_conv3x3(in, hW, hOut, split, H, H, W); };
      double a, b; run_concurrent(split < H, split > 0, cpu, gpu, a, b);
      tc += a; tg += b;
    };
    convStage(dKx, dSx, sx);
    convStage(dKy, dSy, sy);
    // magnitude (split elements); sx,sy GPU rows already on device.
    auto gpuM = [&]() {
      if (esplit <= 0) return;
      int blk = 256, grd = (esplit + blk - 1) / blk;
      mag_kernel<<<grd, blk, 0, s>>>(dSx, dSy, dMag, esplit, 1);
      CUDA_CHECK(cudaMemcpyAsync(mag, dMag, (size_t)esplit * sizeof(float),
                                 cudaMemcpyDeviceToHost, s));
      CUDA_CHECK(cudaStreamSynchronize(s));
    };
    auto cpuM = [&]() { if (esplit < H * W) cpu_mag(sx, sy, mag, esplit, H * W); };
    run_concurrent(esplit < H * W, esplit > 0, cpuM, gpuM, c1, g1); tc += c1; tg += g1;

    double tt = ms_since(t0);
    if (r > 0) { tot.push_back(tt); tcv.push_back(tc); tgv.push_back(tg); }
  }
  double chk = 0; for (int i = 0; i < H * W; i += 9973) chk += mag[i];
  Result res{median(tot), median(tcv), median(tgv), chk};
  emit_csv("sbf", f, res);
  return 0;
}
