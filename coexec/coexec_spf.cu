/*
 * Spam filter: sequential SGD logistic regression.
 * Sizes (HeteroBench): NUM_FEATURES=10000, NUM_TRAINING=4500, EPOCHS=5.
 * The sample loop is inherently sequential (theta carries across samples), so we
 * split the FEATURE dimension: GPU owns theta[0,splitF), CPU owns theta[splitF,NF).
 * Each sample exchanges only the two partial dot-products (a scalar), then both
 * sides update their own theta half. This is the fine-grained / CPU-favored case.
 *
 * Usage: ./coexec_spf <frac> <reps>  -> CSV: spf,frac,t_total,t_cpu,t_gpu,checksum
 */
#include "coexec_common.h"

static const int NF = 10000, NTRAIN = 4500, EPOCHS = 5;
static const float STEP = 0.6f;  // scaled for synthetic (unnormalised) features

__global__ void dot_partial_k(const float *theta, const float *x, int nf, double *out) {
  __shared__ double sh[256];
  int t = threadIdx.x, i = blockIdx.x * blockDim.x + t;
  double v = (i < nf) ? (double)theta[i] * x[i] : 0.0;
  sh[t] = v; __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) { if (t < s) sh[t] += sh[t + s]; __syncthreads(); }
  if (t == 0) atomicAdd(out, sh[0]);
}
__global__ void update_k(float *theta, const float *x, int nf, float coef) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < nf) theta[i] -= STEP * coef * x[i];
}

int main(int argc, char **argv) {
  double f = (argc > 1) ? atof(argv[1]) : 0.5;
  int reps = (argc > 2) ? atoi(argv[2]) : 3;
  int splitF = gpu_split(f, NF);          // GPU owns features [0,splitF)
  int cpuF = NF - splitF;

  float *theta, *data; signed char *label;
  CUDA_CHECK(cudaMallocHost(&theta, NF * sizeof(float)));
  data = (float *)malloc((size_t)NTRAIN * NF * sizeof(float));
  label = (signed char *)malloc(NTRAIN);
  for (size_t i = 0; i < (size_t)NTRAIN * NF; ++i) data[i] = (float)((i % 97) - 48) * 0.01f;
  for (int i = 0; i < NTRAIN; ++i) label[i] = (i % 2);

  float *dTheta, *dData; double *dPartial;
  CUDA_CHECK(cudaMalloc(&dTheta, NF * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dData, (size_t)NTRAIN * NF * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dPartial, sizeof(double)));
  CUDA_CHECK(cudaMemcpy(dData, data, (size_t)NTRAIN * NF * sizeof(float), cudaMemcpyHostToDevice));
  cudaStream_t s; CUDA_CHECK(cudaStreamCreate(&s));
  int blk = 256, grd = (splitF + blk - 1) / blk;

  std::vector<double> tot, tcv, tgv;
  for (int r = 0; r < reps + 1; ++r) {
    memset(theta, 0, NF * sizeof(float));
    CUDA_CHECK(cudaMemset(dTheta, 0, NF * sizeof(float)));
    double tc = 0, tg = 0;
    auto t0 = clk::now();
    for (int e = 0; e < EPOCHS; ++e)
      for (int i = 0; i < NTRAIN; ++i) {
        const float *xh = data + (size_t)i * NF;
        const float *xd = dData + (size_t)i * NF;
        double pg = 0, pc = 0, a, b;
        // Stage 1: partial dot products (concurrent)
        run_concurrent(cpuF > 0, splitF > 0,
          [&] { double acc = 0; for (int k = splitF; k < NF; ++k) acc += (double)theta[k] * xh[k]; pc = acc; },
          [&] { CUDA_CHECK(cudaMemsetAsync(dPartial, 0, sizeof(double), s));
                dot_partial_k<<<grd, blk, 0, s>>>(dTheta, xd, splitF, dPartial);
                CUDA_CHECK(cudaMemcpyAsync(&pg, dPartial, sizeof(double), cudaMemcpyDeviceToHost, s));
                CUDA_CHECK(cudaStreamSynchronize(s)); }, a, b);
        tc += a; tg += b;
        double dot = pg + pc;
        float prob = 1.0f / (1.0f + expf(-(float)dot));
        float coef = prob - (float)label[i];
        // Stage 2: update own theta half (concurrent)
        run_concurrent(cpuF > 0, splitF > 0,
          [&] { for (int k = splitF; k < NF; ++k) theta[k] -= STEP * coef * xh[k]; },
          [&] { update_k<<<grd, blk, 0, s>>>(dTheta, xd, splitF, coef);
                CUDA_CHECK(cudaStreamSynchronize(s)); }, a, b);
        tc += a; tg += b;
      }
    double tt = ms_since(t0);
    if (r > 0) { tot.push_back(tt); tcv.push_back(tc); tgv.push_back(tg); }
  }
  if (splitF > 0) CUDA_CHECK(cudaMemcpy(theta, dTheta, splitF * sizeof(float), cudaMemcpyDeviceToHost));
  double chk = 0; for (int k = 0; k < NF; k += 311) chk += theta[k];
  Result res{median(tot), median(tcv), median(tgv), chk};
  emit_csv("spf", f, res);
  free(data); free(label);
  return 0;
}
