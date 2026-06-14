/*
 * CNN inference: conv2d(3x3) -> relu -> max_pool(2x2) -> flatten -> FC -> softmax.
 * Sizes (HeteroBench): image 1024x2048; pool 512x1024; FC 524288->2048.
 * Two co-executed kernels: conv2d splits output ROWS; the (memory-bound) FC
 * splits output NEURONS. relu/pool/softmax run on the host (cheap).
 *
 * Usage: ./coexec_cnn <frac> <reps>  -> CSV: cnn,frac,t_total,t_cpu,t_gpu,checksum
 */
#include "coexec_common.h"

static const int IH = 1024, IW = 2048;       // conv output = image size (pad=1,stride=1)
static const int PH = 512, PW = 1024;         // pool output
static const int FK = PH * PW;                // 524288 flatten / FC input
static const int FN = 2048;                   // FC output neurons

__global__ void conv_kernel(const float *in, const float *ker, float *out,
                            int r0, int rows, int H, int W, float bias) {
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  int ii = blockIdx.y * blockDim.y + threadIdx.y;  // 0..rows-1
  if (ii >= rows || j >= W) return;
  int i = r0 + ii;
  float acc = bias;
  for (int ki = 0; ki < 3; ++ki)
    for (int kj = 0; kj < 3; ++kj) {
      int yy = i + ki - 1, xx = j + kj - 1;
      if (yy >= 0 && yy < H && xx >= 0 && xx < W)
        acc += in[(size_t)yy * W + xx] * ker[ki * 3 + kj];
    }
  out[(size_t)i * W + j] = acc;
}

static void cpu_conv(const float *in, const float *ker, float *out, int r0,
                     int r1, int H, int W, float bias) {
#pragma omp parallel for schedule(static)
  for (int i = r0; i < r1; ++i)
    for (int j = 0; j < W; ++j) {
      float acc = bias;
      for (int ki = 0; ki < 3; ++ki)
        for (int kj = 0; kj < 3; ++kj) {
          int yy = i + ki - 1, xx = j + kj - 1;
          if (yy >= 0 && yy < H && xx >= 0 && xx < W)
            acc += in[(size_t)yy * W + xx] * ker[ki * 3 + kj];
        }
      out[(size_t)i * W + j] = acc;
    }
}

// FC: y[n] = sum_k flat[k]*Wt[n][k] + b[n], Wt neuron-major [FN][FK].
__global__ void fc_kernel(const float *flat, const float *Wt, const float *b,
                          float *y, int n0, int ncols) {
  int nn = blockIdx.x * blockDim.x + threadIdx.x;  // 0..ncols-1
  if (nn >= ncols) return;
  int n = n0 + nn;
  const float *Wn = Wt + (size_t)n * FK;
  float acc = 0.0f;
  for (int k = 0; k < FK; ++k) acc += flat[k] * Wn[k];
  y[n] = acc + b[n];
}

static void cpu_fc(const float *flat, const float *Wt, const float *b, float *y,
                   int n0, int n1) {
#pragma omp parallel for schedule(static)
  for (int n = n0; n < n1; ++n) {
    const float *Wn = Wt + (size_t)n * FK;
    float acc = 0.0f;
    for (int k = 0; k < FK; ++k) acc += flat[k] * Wn[k];
    y[n] = acc + b[n];
  }
}

int main(int argc, char **argv) {
  double f = (argc > 1) ? atof(argv[1]) : 0.5;
  int reps = (argc > 2) ? atoi(argv[2]) : 2;

  float ker[9] = {0.1f, 0.1f, 0.1f, 0.1f, 0.2f, 0.1f, 0.1f, 0.1f, 0.1f};
  float *in, *conv, *flat, *y, *bfc;
  CUDA_CHECK(cudaMallocHost(&in, (size_t)IH * IW * sizeof(float)));
  CUDA_CHECK(cudaMallocHost(&conv, (size_t)IH * IW * sizeof(float)));
  CUDA_CHECK(cudaMallocHost(&flat, (size_t)FK * sizeof(float)));
  CUDA_CHECK(cudaMallocHost(&y, (size_t)FN * sizeof(float)));
  CUDA_CHECK(cudaMallocHost(&bfc, (size_t)FN * sizeof(float)));
  float *Wt = (float *)malloc((size_t)FN * FK * sizeof(float));  // 4.3 GB pageable
  for (size_t i = 0; i < (size_t)IH * IW; ++i) in[i] = (float)((i % 13) - 6) * 0.02f;
  for (int n = 0; n < FN; ++n) bfc[n] = (float)((n % 9) - 4) * 0.01f;
  for (size_t i = 0; i < (size_t)FN * FK; ++i) Wt[i] = (float)((i % 17) - 8) * 0.0002f;

  float *dIn, *dKer, *dConv, *dFlat, *dWt, *dB, *dY;
  CUDA_CHECK(cudaMalloc(&dIn, (size_t)IH * IW * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dKer, 9 * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dConv, (size_t)IH * IW * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dFlat, (size_t)FK * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dWt, (size_t)FN * FK * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dB, (size_t)FN * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dY, (size_t)FN * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(dIn, in, (size_t)IH * IW * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dKer, ker, 9 * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dWt, Wt, (size_t)FN * FK * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dB, bfc, (size_t)FN * sizeof(float), cudaMemcpyHostToDevice));
  cudaStream_t s; CUDA_CHECK(cudaStreamCreate(&s));

  int splitR = gpu_split(f, IH);   // conv rows on GPU
  int splitN = gpu_split(f, FN);   // FC neurons on GPU

  std::vector<double> tot, tcv, tgv;
  for (int r = 0; r < reps + 1; ++r) {
    double tc = 0, tg = 0, c1, g1;
    auto t0 = clk::now();
    // Stage 1: conv2d (split rows). Full input resident on device + host.
    auto gpuC = [&]() {
      if (splitR <= 0) return;
      dim3 b(16, 16), g((IW + 15) / 16, (splitR + 15) / 16);
      conv_kernel<<<g, b, 0, s>>>(dIn, dKer, dConv, 0, splitR, IH, IW, 0.1f);
      CUDA_CHECK(cudaMemcpyAsync(conv, dConv, (size_t)splitR * IW * sizeof(float),
                                 cudaMemcpyDeviceToHost, s));
      CUDA_CHECK(cudaStreamSynchronize(s));
    };
    auto cpuC = [&]() { if (splitR < IH) cpu_conv(in, ker, conv, splitR, IH, IH, IW, 0.1f); };
    run_concurrent(splitR < IH, splitR > 0, cpuC, gpuC, c1, g1); tc += c1; tg += g1;

    // Stage 2: relu + max-pool + flatten (host, cheap)
    auto th0 = clk::now();
#pragma omp parallel for schedule(static)
    for (int pi = 0; pi < PH; ++pi)
      for (int pj = 0; pj < PW; ++pj) {
        float mx = -1e30f;
        for (int di = 0; di < 2; ++di)
          for (int dj = 0; dj < 2; ++dj) {
            float v = conv[(size_t)(pi * 2 + di) * IW + (pj * 2 + dj)];
            v = v > 0 ? v : 0;  // relu
            mx = v > mx ? v : mx;
          }
        flat[(size_t)pi * PW + pj] = mx;
      }
    tc += ms_since(th0);

    // Stage 3: FC (split neurons). flat -> device.
    CUDA_CHECK(cudaMemcpy(dFlat, flat, (size_t)FK * sizeof(float), cudaMemcpyHostToDevice));
    auto gpuF = [&]() {
      if (splitN <= 0) return;
      int blk = 128, grd = (splitN + blk - 1) / blk;
      fc_kernel<<<grd, blk, 0, s>>>(dFlat, dWt, dB, dY, 0, splitN);
      CUDA_CHECK(cudaMemcpyAsync(y, dY, (size_t)splitN * sizeof(float),
                                 cudaMemcpyDeviceToHost, s));
      CUDA_CHECK(cudaStreamSynchronize(s));
    };
    auto cpuF = [&]() { if (splitN < FN) cpu_fc(flat, Wt, bfc, y, splitN, FN); };
    run_concurrent(splitN < FN, splitN > 0, cpuF, gpuF, c1, g1); tc += c1; tg += g1;

    // Stage 4: softmax over FN (host, cheap)
    auto ts0 = clk::now();
    double mx = y[0]; for (int n = 1; n < FN; ++n) mx = fmax(mx, (double)y[n]);
    double sum = 0; for (int n = 0; n < FN; ++n) sum += exp(y[n] - mx);
    for (int n = 0; n < FN; ++n) y[n] = (float)(exp(y[n] - mx) / sum * 1e4);
    tc += ms_since(ts0);

    double tt = ms_since(t0);
    if (r > 0) { tot.push_back(tt); tcv.push_back(tc); tgv.push_back(tg); }
  }
  double chk = 0; for (int n = 0; n < FN; n += 37) chk += y[n];
  Result res{median(tot), median(tcv), median(tgv), chk};
  emit_csv("cnn", f, res);
  free(Wt);
  return 0;
}
