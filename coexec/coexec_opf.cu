/*
 * Optical flow (Lucas-Kanade structure-tensor pipeline):
 *   gradients (x,y,z) -> 7-tap separable smoothing -> outer product (6 tensor
 *   components) -> 3-tap separable smoothing -> 2x2 flow solve.
 * Size (HeteroBench 4K): 2160 x 3840, 5 input frames.
 * Every stage splits image ROWS between GPU [0,split) and CPU [split,H); the
 * full previous output is resident on the device (CPU rows pushed each stage)
 * so vertical-convolution halos are satisfied on both sides.
 *
 * Usage: ./coexec_opf <frac> <reps>  -> CSV: opf,frac,t_total,t_cpu,t_gpu,checksum
 */
#include "coexec_common.h"

static const int H = 2160, W = 3840;
static const float GW[5] = {0.0833333f, -0.6666667f, 0.0f, 0.6666667f, -0.0833333f}; // {1,-8,0,8,-1}/12
static const float GF[7] = {0.0755f, 0.133f, 0.1869f, 0.2903f, 0.1869f, 0.133f, 0.0755f};
static const float TF[3] = {0.3243f, 0.3513f, 0.3243f};

__global__ void vconv_k(const float *in, float *out, int rows, const float *t, int nt) {
  int j = blockIdx.x * blockDim.x + threadIdx.x, i = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= rows || j >= W) return;
  int half = nt / 2; float acc = 0;
  for (int k = 0; k < nt; ++k) { int yy = i + k - half; yy = yy < 0 ? 0 : (yy >= H ? H - 1 : yy); acc += in[(size_t)yy * W + j] * t[k]; }
  out[(size_t)i * W + j] = acc;
}
__global__ void hconv_k(const float *in, float *out, int rows, const float *t, int nt) {
  int j = blockIdx.x * blockDim.x + threadIdx.x, i = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= rows || j >= W) return;
  int half = nt / 2; float acc = 0;
  for (int k = 0; k < nt; ++k) { int xx = j + k - half; xx = xx < 0 ? 0 : (xx >= W ? W - 1 : xx); acc += in[(size_t)i * W + xx] * t[k]; }
  out[(size_t)i * W + j] = acc;
}
__global__ void gradz_k(const float *f0, const float *f1, const float *f2, const float *f3, const float *f4, float *gz, int rows, const float *w) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x; if (idx >= rows * W) return;
  gz[idx] = f0[idx]*w[0]+f1[idx]*w[1]+f2[idx]*w[2]+f3[idx]*w[3]+f4[idx]*w[4];
}
__global__ void outer_k(const float *gx, const float *gy, const float *gz, float *t0, float *t1, float *t2, float *t3, float *t4, float *t5, int rows) {
  int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= rows * W) return;
  float x = gx[i], y = gy[i], z = gz[i];
  t0[i]=x*x; t1[i]=y*y; t2[i]=z*z; t3[i]=x*y; t4[i]=x*z; t5[i]=y*z;
}
__global__ void flow_k(const float *t0, const float *t1, const float *t3, const float *t4, const float *t5, float *vx, float *vy, int rows) {
  int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= rows * W) return;
  float d = t0[i]*t1[i]-t3[i]*t3[i]; if (fabsf(d) < 1e-8f) d = 1e-8f;
  vx[i] = (t5[i]*t3[i]-t4[i]*t1[i])/d; vy[i] = (t4[i]*t3[i]-t5[i]*t0[i])/d;
}

// CPU separable conv
static void cpu_vconv(const float *in, float *out, int r0, int r1, const float *t, int nt) {
  int half = nt / 2;
#pragma omp parallel for schedule(static)
  for (int i = r0; i < r1; ++i) for (int j = 0; j < W; ++j) { float acc=0; for(int k=0;k<nt;++k){int yy=i+k-half;yy=yy<0?0:(yy>=H?H-1:yy);acc+=in[(size_t)yy*W+j]*t[k];} out[(size_t)i*W+j]=acc; }
}
static void cpu_hconv(const float *in, float *out, int r0, int r1, const float *t, int nt) {
  int half = nt / 2;
#pragma omp parallel for schedule(static)
  for (int i = r0; i < r1; ++i) for (int j = 0; j < W; ++j) { float acc=0; for(int k=0;k<nt;++k){int xx=j+k-half;xx=xx<0?0:(xx>=W?W-1:xx);acc+=in[(size_t)i*W+xx]*t[k];} out[(size_t)i*W+j]=acc; }
}

int main(int argc, char **argv) {
  double f = (argc > 1) ? atof(argv[1]) : 0.5;
  int reps = (argc > 2) ? atoi(argv[2]) : 2;
  int split = gpu_split(f, H); size_t NP = (size_t)H * W;

  // host buffers
  float *fr[5], *gx, *gy, *gz, *tg0, *tg1, *tg2, *t[6], *tt[6], *vx, *vy;
  auto hostA = [&](float **p){ CUDA_CHECK(cudaMallocHost(p, NP*sizeof(float))); };
  for (int k=0;k<5;++k) hostA(&fr[k]);
  hostA(&gx);hostA(&gy);hostA(&gz);hostA(&tg0);hostA(&tg1);hostA(&tg2);
  for(int k=0;k<6;++k){hostA(&t[k]);hostA(&tt[k]);} hostA(&vx);hostA(&vy);
  for (int k=0;k<5;++k) for (size_t i=0;i<NP;++i) fr[k][i]=(float)((i+k*7)%256)/256.f;

  // device buffers
  float *dfr[5], *dgx,*dgy,*dgz,*dtg0,*dtg1,*dtg2,*dt[6],*dtt[6],*dvx,*dvy,*dGW,*dGF,*dTF;
  auto devA=[&](float **p){ CUDA_CHECK(cudaMalloc(p, NP*sizeof(float))); };
  for(int k=0;k<5;++k){devA(&dfr[k]); CUDA_CHECK(cudaMemcpy(dfr[k],fr[k],NP*sizeof(float),cudaMemcpyHostToDevice));}
  devA(&dgx);devA(&dgy);devA(&dgz);devA(&dtg0);devA(&dtg1);devA(&dtg2);
  for(int k=0;k<6;++k){devA(&dt[k]);devA(&dtt[k]);} devA(&dvx);devA(&dvy);
  CUDA_CHECK(cudaMalloc(&dGW,5*sizeof(float))); CUDA_CHECK(cudaMemcpy(dGW,GW,5*sizeof(float),cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMalloc(&dGF,7*sizeof(float))); CUDA_CHECK(cudaMemcpy(dGF,GF,7*sizeof(float),cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMalloc(&dTF,3*sizeof(float))); CUDA_CHECK(cudaMemcpy(dTF,TF,3*sizeof(float),cudaMemcpyHostToDevice));
  cudaStream_t s; CUDA_CHECK(cudaStreamCreate(&s));
  dim3 B(16,16), G((W+15)/16,(split+15)/16);
  int eblk=256, egrd=(split*W+eblk-1)/eblk;

  auto pull=[&](float*h,float*d){ if(split>0) CUDA_CHECK(cudaMemcpyAsync(h,d,(size_t)split*W*sizeof(float),cudaMemcpyDeviceToHost,s)); };
  auto push=[&](float*d,float*h){ if(split<H) CUDA_CHECK(cudaMemcpy(d+(size_t)split*W,h+(size_t)split*W,(size_t)(H-split)*W*sizeof(float),cudaMemcpyHostToDevice)); };

  std::vector<double> tot, tcv, tgv;
  for (int r = 0; r < reps + 1; ++r) {
    double tc=0,tg=0,a,b;
    auto t0=clk::now();
    // Stage 1: gx=hconv(frame2,GW), gy=vconv(frame2,GW), gz=pointwise
    run_concurrent(split<H,split>0,
      [&]{ if(split<H){cpu_hconv(fr[2],gx,split,H,GW,5);cpu_vconv(fr[2],gy,split,H,GW,5);
        for(int i=split;i<H;++i)for(int j=0;j<W;++j){size_t p=(size_t)i*W+j;gz[p]=fr[0][p]*GW[0]+fr[1][p]*GW[1]+fr[2][p]*GW[2]+fr[3][p]*GW[3]+fr[4][p]*GW[4];}} },
      [&]{ if(split>0){hconv_k<<<G,B,0,s>>>(dfr[2],dgx,split,dGW,5);vconv_k<<<G,B,0,s>>>(dfr[2],dgy,split,dGW,5);gradz_k<<<egrd,eblk,0,s>>>(dfr[0],dfr[1],dfr[2],dfr[3],dfr[4],dgz,split,dGW);pull(gx,dgx);pull(gy,dgy);pull(gz,dgz);CUDA_CHECK(cudaStreamSynchronize(s));} },a,b);tc+=a;tg+=b;
    push(dgx,gx);push(dgy,gy);push(dgz,gz);
    // Stage 2: 7-tap vertical smoothing of gx,gy,gz -> tg
    run_concurrent(split<H,split>0,
      [&]{ if(split<H){cpu_vconv(gx,tg0,split,H,GF,7);cpu_vconv(gy,tg1,split,H,GF,7);cpu_vconv(gz,tg2,split,H,GF,7);} },
      [&]{ if(split>0){vconv_k<<<G,B,0,s>>>(dgx,dtg0,split,dGF,7);vconv_k<<<G,B,0,s>>>(dgy,dtg1,split,dGF,7);vconv_k<<<G,B,0,s>>>(dgz,dtg2,split,dGF,7);pull(tg0,dtg0);pull(tg1,dtg1);pull(tg2,dtg2);CUDA_CHECK(cudaStreamSynchronize(s));} },a,b);tc+=a;tg+=b;
    push(dtg0,tg0);push(dtg1,tg1);push(dtg2,tg2);
    // Stage 3: 7-tap horizontal smoothing -> gx,gy,gz (reuse)
    run_concurrent(split<H,split>0,
      [&]{ if(split<H){cpu_hconv(tg0,gx,split,H,GF,7);cpu_hconv(tg1,gy,split,H,GF,7);cpu_hconv(tg2,gz,split,H,GF,7);} },
      [&]{ if(split>0){hconv_k<<<G,B,0,s>>>(dtg0,dgx,split,dGF,7);hconv_k<<<G,B,0,s>>>(dtg1,dgy,split,dGF,7);hconv_k<<<G,B,0,s>>>(dtg2,dgz,split,dGF,7);pull(gx,dgx);pull(gy,dgy);pull(gz,dgz);CUDA_CHECK(cudaStreamSynchronize(s));} },a,b);tc+=a;tg+=b;
    push(dgx,gx);push(dgy,gy);push(dgz,gz);
    // Stage 4: outer product -> t[0..5]
    run_concurrent(split<H,split>0,
      [&]{ if(split<H){
#pragma omp parallel for schedule(static)
        for(int i=split;i<H;++i)for(int j=0;j<W;++j){size_t p=(size_t)i*W+j;float x=gx[p],y=gy[p],z=gz[p];t[0][p]=x*x;t[1][p]=y*y;t[2][p]=z*z;t[3][p]=x*y;t[4][p]=x*z;t[5][p]=y*z;} } },
      [&]{ if(split>0){outer_k<<<egrd,eblk,0,s>>>(dgx,dgy,dgz,dt[0],dt[1],dt[2],dt[3],dt[4],dt[5],split);for(int k=0;k<6;++k)pull(t[k],dt[k]);CUDA_CHECK(cudaStreamSynchronize(s));} },a,b);tc+=a;tg+=b;
    for(int k=0;k<6;++k)push(dt[k],t[k]);
    // Stage 5: 3-tap vertical smoothing of tensor -> tt
    run_concurrent(split<H,split>0,
      [&]{ if(split<H)for(int k=0;k<6;++k)cpu_vconv(t[k],tt[k],split,H,TF,3); },
      [&]{ if(split>0){for(int k=0;k<6;++k)vconv_k<<<G,B,0,s>>>(dt[k],dtt[k],split,dTF,3);for(int k=0;k<6;++k)pull(tt[k],dtt[k]);CUDA_CHECK(cudaStreamSynchronize(s));} },a,b);tc+=a;tg+=b;
    for(int k=0;k<6;++k)push(dtt[k],tt[k]);
    // Stage 6: 3-tap horizontal smoothing -> t
    run_concurrent(split<H,split>0,
      [&]{ if(split<H)for(int k=0;k<6;++k)cpu_hconv(tt[k],t[k],split,H,TF,3); },
      [&]{ if(split>0){for(int k=0;k<6;++k)hconv_k<<<G,B,0,s>>>(dtt[k],dt[k],split,dTF,3);for(int k=0;k<6;++k)pull(t[k],dt[k]);CUDA_CHECK(cudaStreamSynchronize(s));} },a,b);tc+=a;tg+=b;
    for(int k=0;k<6;++k)push(dt[k],t[k]);
    // Stage 7: flow calc -> vx,vy
    run_concurrent(split<H,split>0,
      [&]{ if(split<H){
#pragma omp parallel for schedule(static)
        for(int i=split;i<H;++i)for(int j=0;j<W;++j){size_t p=(size_t)i*W+j;float d=t[0][p]*t[1][p]-t[3][p]*t[3][p];if(fabsf(d)<1e-8f)d=1e-8f;vx[p]=(t[5][p]*t[3][p]-t[4][p]*t[1][p])/d;vy[p]=(t[4][p]*t[3][p]-t[5][p]*t[0][p])/d;} } },
      [&]{ if(split>0){flow_k<<<egrd,eblk,0,s>>>(dt[0],dt[1],dt[3],dt[4],dt[5],dvx,dvy,split);pull(vx,dvx);pull(vy,dvy);CUDA_CHECK(cudaStreamSynchronize(s));} },a,b);tc+=a;tg+=b;

    double ttm = ms_since(t0);
    if (r > 0) { tot.push_back(ttm); tcv.push_back(tc); tgv.push_back(tg); }
  }
  double chk = 0; for (size_t i = 0; i < NP; i += 99991) chk += vx[i] + vy[i];
  Result res{median(tot), median(tcv), median(tgv), chk};
  emit_csv("opf", f, res);
  return 0;
}
