/*
 * Canny edge detection: gaussian -> gradient(intensity+direction) ->
 * non-max suppression (edge thinning) -> double threshold -> hysteresis.
 * Size (HeteroBench stanford image): 1919 x 1439.
 * Each stage splits image ROWS between GPU [0,split) and CPU [split,H); since
 * every stage reads neighbour rows, after each stage the CPU-computed rows are
 * pushed to the device so both halves see the full previous output.
 *
 * Usage: ./coexec_ced <frac> <reps>  -> CSV: ced,frac,t_total,t_cpu,t_gpu,checksum
 */
#include "coexec_common.h"

static const int H = 1919, W = 1439;
__constant__ float GK[9] = {0.0625f, 0.125f, 0.0625f, 0.125f, 0.25f,
                            0.125f, 0.0625f, 0.125f, 0.0625f};

__device__ __host__ static inline int quantize(float gx, float gy) {
  float a = atan2f(gy, gx) * 57.2957795f;
  if (a < 0) a += 180.0f;
  if (a < 22.5f || a >= 157.5f) return 0;
  if (a < 67.5f) return 1;
  if (a < 112.5f) return 2;
  return 3;
}

__global__ void gauss_k(const float *in, float *out, int rows) {
  int j = blockIdx.x * blockDim.x + threadIdx.x, i = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= rows || j >= W) return;
  if (i == 0 || i == H - 1 || j == 0 || j == W - 1) { out[(size_t)i * W + j] = in[(size_t)i * W + j]; return; }
  float acc = 0;
  for (int ki = -1; ki <= 1; ++ki) for (int kj = -1; kj <= 1; ++kj)
    acc += in[(size_t)(i + ki) * W + (j + kj)] * GK[(ki + 1) * 3 + (kj + 1)];
  out[(size_t)i * W + j] = acc;
}
__global__ void grad_k(const float *g, float *inten, int *dir, int rows) {
  int j = blockIdx.x * blockDim.x + threadIdx.x, i = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= rows || j >= W) return;
  if (i == 0 || i == H - 1 || j == 0 || j == W - 1) { inten[(size_t)i * W + j] = 0; dir[(size_t)i * W + j] = 0; return; }
  float gx = -g[(size_t)(i-1)*W+j-1] + g[(size_t)(i-1)*W+j+1] - 2*g[(size_t)i*W+j-1] + 2*g[(size_t)i*W+j+1] - g[(size_t)(i+1)*W+j-1] + g[(size_t)(i+1)*W+j+1];
  float gy = -g[(size_t)(i-1)*W+j-1] - 2*g[(size_t)(i-1)*W+j] - g[(size_t)(i-1)*W+j+1] + g[(size_t)(i+1)*W+j-1] + 2*g[(size_t)(i+1)*W+j] + g[(size_t)(i+1)*W+j+1];
  inten[(size_t)i * W + j] = sqrtf(gx * gx + gy * gy);
  dir[(size_t)i * W + j] = quantize(gx, gy);
}
__global__ void thin_k(const float *inten, const int *dir, float *out, int rows) {
  int j = blockIdx.x * blockDim.x + threadIdx.x, i = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= rows || j >= W) return;
  if (i == 0 || i == H - 1 || j == 0 || j == W - 1) { out[(size_t)i * W + j] = 0; return; }
  float v = inten[(size_t)i * W + j]; int d = dir[(size_t)i * W + j]; float a, b;
  if (d == 0) { a = inten[(size_t)i*W+j-1]; b = inten[(size_t)i*W+j+1]; }
  else if (d == 1) { a = inten[(size_t)(i-1)*W+j+1]; b = inten[(size_t)(i+1)*W+j-1]; }
  else if (d == 2) { a = inten[(size_t)(i-1)*W+j]; b = inten[(size_t)(i+1)*W+j]; }
  else { a = inten[(size_t)(i-1)*W+j-1]; b = inten[(size_t)(i+1)*W+j+1]; }
  out[(size_t)i * W + j] = (v >= a && v >= b) ? v : 0.0f;
}
__global__ void thresh_k(const float *in, unsigned char *out, int rows, float lo, float hi) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= rows * W) return;
  float v = in[idx]; out[idx] = v > hi ? 255 : (v > lo ? 100 : 0);
}
__global__ void hyst_k(const unsigned char *in, unsigned char *out, int rows) {
  int j = blockIdx.x * blockDim.x + threadIdx.x, i = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= rows || j >= W) return;
  unsigned char v = in[(size_t)i * W + j];
  if (v == 255) { out[(size_t)i * W + j] = 255; return; }
  if (v == 100 && i > 0 && i < H - 1 && j > 0 && j < W - 1) {
    int strong = 0;
    for (int ki = -1; ki <= 1; ++ki) for (int kj = -1; kj <= 1; ++kj)
      if (in[(size_t)(i + ki) * W + (j + kj)] == 255) strong = 1;
    out[(size_t)i * W + j] = strong ? 255 : 0;
  } else out[(size_t)i * W + j] = 0;
}

// ---- CPU mirrors ----
static void cpu_gauss(const float *in, float *out, int r0, int r1) {
  const float gk[9] = {0.0625f, 0.125f, 0.0625f, 0.125f, 0.25f, 0.125f, 0.0625f, 0.125f, 0.0625f};
#pragma omp parallel for schedule(static)
  for (int i = r0; i < r1; ++i) for (int j = 0; j < W; ++j) {
    if (i==0||i==H-1||j==0||j==W-1){out[(size_t)i*W+j]=in[(size_t)i*W+j];continue;}
    float acc = 0;
    for (int ki=-1;ki<=1;++ki) for(int kj=-1;kj<=1;++kj) acc+=in[(size_t)(i+ki)*W+(j+kj)]*gk[(ki+1)*3+(kj+1)];
    out[(size_t)i*W+j]=acc;
  }
}
static void cpu_grad(const float *g, float *inten, int *dir, int r0, int r1) {
#pragma omp parallel for schedule(static)
  for (int i = r0; i < r1; ++i) for (int j = 0; j < W; ++j) {
    if (i==0||i==H-1||j==0||j==W-1){inten[(size_t)i*W+j]=0;dir[(size_t)i*W+j]=0;continue;}
    float gx=-g[(size_t)(i-1)*W+j-1]+g[(size_t)(i-1)*W+j+1]-2*g[(size_t)i*W+j-1]+2*g[(size_t)i*W+j+1]-g[(size_t)(i+1)*W+j-1]+g[(size_t)(i+1)*W+j+1];
    float gy=-g[(size_t)(i-1)*W+j-1]-2*g[(size_t)(i-1)*W+j]-g[(size_t)(i-1)*W+j+1]+g[(size_t)(i+1)*W+j-1]+2*g[(size_t)(i+1)*W+j]+g[(size_t)(i+1)*W+j+1];
    inten[(size_t)i*W+j]=sqrtf(gx*gx+gy*gy); dir[(size_t)i*W+j]=quantize(gx,gy);
  }
}
static void cpu_thin(const float *inten, const int *dir, float *out, int r0, int r1) {
#pragma omp parallel for schedule(static)
  for (int i = r0; i < r1; ++i) for (int j = 0; j < W; ++j) {
    if (i==0||i==H-1||j==0||j==W-1){out[(size_t)i*W+j]=0;continue;}
    float v=inten[(size_t)i*W+j]; int d=dir[(size_t)i*W+j]; float a,b;
    if(d==0){a=inten[(size_t)i*W+j-1];b=inten[(size_t)i*W+j+1];}
    else if(d==1){a=inten[(size_t)(i-1)*W+j+1];b=inten[(size_t)(i+1)*W+j-1];}
    else if(d==2){a=inten[(size_t)(i-1)*W+j];b=inten[(size_t)(i+1)*W+j];}
    else{a=inten[(size_t)(i-1)*W+j-1];b=inten[(size_t)(i+1)*W+j+1];}
    out[(size_t)i*W+j]=(v>=a&&v>=b)?v:0.0f;
  }
}
static void cpu_thresh(const float *in, unsigned char *out, int e0, int e1, float lo, float hi) {
#pragma omp parallel for schedule(static)
  for (int i = e0; i < e1; ++i) { float v=in[i]; out[i]=v>hi?255:(v>lo?100:0); }
}
static void cpu_hyst(const unsigned char *in, unsigned char *out, int r0, int r1) {
#pragma omp parallel for schedule(static)
  for (int i = r0; i < r1; ++i) for (int j = 0; j < W; ++j) {
    unsigned char v=in[(size_t)i*W+j];
    if(v==255){out[(size_t)i*W+j]=255;continue;}
    if(v==100&&i>0&&i<H-1&&j>0&&j<W-1){int st=0;for(int ki=-1;ki<=1;++ki)for(int kj=-1;kj<=1;++kj)if(in[(size_t)(i+ki)*W+(j+kj)]==255)st=1;out[(size_t)i*W+j]=st?255:0;}
    else out[(size_t)i*W+j]=0;
  }
}

int main(int argc, char **argv) {
  double f = (argc > 1) ? atof(argv[1]) : 0.5;
  int reps = (argc > 2) ? atoi(argv[2]) : 3;
  int split = gpu_split(f, H), esplit = split * W;
  size_t NP = (size_t)H * W;

  float *in, *g, *inten; int *dir; unsigned char *thr, *out;
  float *thin;
  CUDA_CHECK(cudaMallocHost(&in, NP*sizeof(float)));
  CUDA_CHECK(cudaMallocHost(&g, NP*sizeof(float)));
  CUDA_CHECK(cudaMallocHost(&inten, NP*sizeof(float)));
  CUDA_CHECK(cudaMallocHost(&thin, NP*sizeof(float)));
  CUDA_CHECK(cudaMallocHost(&dir, NP*sizeof(int)));
  CUDA_CHECK(cudaMallocHost(&thr, NP));
  CUDA_CHECK(cudaMallocHost(&out, NP));
  for (size_t i = 0; i < NP; ++i) in[i] = (float)(i % 256);

  float *dIn,*dG,*dInten,*dThin; int *dDir; unsigned char *dThr,*dOut;
  CUDA_CHECK(cudaMalloc(&dIn,NP*sizeof(float))); CUDA_CHECK(cudaMalloc(&dG,NP*sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dInten,NP*sizeof(float))); CUDA_CHECK(cudaMalloc(&dThin,NP*sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dDir,NP*sizeof(int))); CUDA_CHECK(cudaMalloc(&dThr,NP)); CUDA_CHECK(cudaMalloc(&dOut,NP));
  CUDA_CHECK(cudaMemcpy(dIn,in,NP*sizeof(float),cudaMemcpyHostToDevice));
  cudaStream_t s; CUDA_CHECK(cudaStreamCreate(&s));
  dim3 B(16,16), G((W+15)/16,(split+15)/16);

  auto push_cpu_rows = [&](void *dst, void *src, size_t elem) {  // sync cpu rows -> device
    if (split < H) CUDA_CHECK(cudaMemcpy((char*)dst+(size_t)split*W*elem,(char*)src+(size_t)split*W*elem,(size_t)(H-split)*W*elem,cudaMemcpyHostToDevice));
  };
  std::vector<double> tot, tcv, tgv;
  for (int r = 0; r < reps + 1; ++r) {
    double tc = 0, tg = 0, a, b;
    auto t0 = clk::now();
    // gaussian
    run_concurrent(split<H, split>0,
      [&]{ if(split<H) cpu_gauss(in,g,split,H); },
      [&]{ if(split>0){ gauss_k<<<G,B,0,s>>>(dIn,dG,split); CUDA_CHECK(cudaMemcpyAsync(g,dG,(size_t)split*W*sizeof(float),cudaMemcpyDeviceToHost,s)); CUDA_CHECK(cudaStreamSynchronize(s)); } }, a,b); tc+=a;tg+=b;
    push_cpu_rows(dG,g,sizeof(float));
    // gradient
    run_concurrent(split<H, split>0,
      [&]{ if(split<H) cpu_grad(g,inten,dir,split,H); },
      [&]{ if(split>0){ grad_k<<<G,B,0,s>>>(dG,dInten,dDir,split); CUDA_CHECK(cudaMemcpyAsync(inten,dInten,(size_t)split*W*sizeof(float),cudaMemcpyDeviceToHost,s)); CUDA_CHECK(cudaMemcpyAsync(dir,dDir,(size_t)split*W*sizeof(int),cudaMemcpyDeviceToHost,s)); CUDA_CHECK(cudaStreamSynchronize(s)); } }, a,b); tc+=a;tg+=b;
    push_cpu_rows(dInten,inten,sizeof(float)); push_cpu_rows(dDir,dir,sizeof(int));
    // thinning
    run_concurrent(split<H, split>0,
      [&]{ if(split<H) cpu_thin(inten,dir,thin,split,H); },
      [&]{ if(split>0){ thin_k<<<G,B,0,s>>>(dInten,dDir,dThin,split); CUDA_CHECK(cudaMemcpyAsync(thin,dThin,(size_t)split*W*sizeof(float),cudaMemcpyDeviceToHost,s)); CUDA_CHECK(cudaStreamSynchronize(s)); } }, a,b); tc+=a;tg+=b;
    push_cpu_rows(dThin,thin,sizeof(float));
    // threshold (elementwise)
    run_concurrent(esplit<H*W, esplit>0,
      [&]{ if(esplit<H*W) cpu_thresh(thin,thr,esplit,H*W,30.f,90.f); },
      [&]{ if(esplit>0){ int blk=256,grd=(esplit+blk-1)/blk; thresh_k<<<grd,blk,0,s>>>(dThin,dThr,split,30.f,90.f); CUDA_CHECK(cudaMemcpyAsync(thr,dThr,(size_t)esplit,cudaMemcpyDeviceToHost,s)); CUDA_CHECK(cudaStreamSynchronize(s)); } }, a,b); tc+=a;tg+=b;
    push_cpu_rows(dThr,thr,1);
    // hysteresis
    run_concurrent(split<H, split>0,
      [&]{ if(split<H) cpu_hyst(thr,out,split,H); },
      [&]{ if(split>0){ hyst_k<<<G,B,0,s>>>(dThr,dOut,split); CUDA_CHECK(cudaMemcpyAsync(out,dOut,(size_t)split*W,cudaMemcpyDeviceToHost,s)); CUDA_CHECK(cudaStreamSynchronize(s)); } }, a,b); tc+=a;tg+=b;

    double tt = ms_since(t0);
    if (r > 0) { tot.push_back(tt); tcv.push_back(tc); tgv.push_back(tg); }
  }
  double chk = 0; for (size_t i = 0; i < NP; i += 9973) chk += out[i];
  Result res{median(tot), median(tcv), median(tgv), chk};
  emit_csv("ced", f, res);
  return 0;
}
