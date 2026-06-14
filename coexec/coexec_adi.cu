/*
 * ADI (PolyBench alternating direction implicit): TSTEPS timesteps, each with a
 * column sweep (Thomas solve along rows) then a row sweep (along columns).
 * Size (HeteroBench): N=1024, TSTEPS=50.
 * Co-execution flips the split axis per phase: the column sweep splits ROWS, the
 * row sweep splits COLUMNS. Between phases the two halves are reconciled
 * (contiguous copy for rows, cudaMemcpy2D for the column-strided region).
 *
 * Usage: ./coexec_adi <frac> <reps>  -> CSV: adi,frac,t_total,t_cpu,t_gpu,checksum
 */
#include "coexec_common.h"

static const int N = 1024, TSTEPS = 50;

__global__ void col_sweep_k(double *X, const double *A, double *B, int split) {
  int c2 = blockIdx.x * blockDim.x + threadIdx.x;
  if (c2 >= split) return;
  double *Xr = X + (size_t)c2 * N; const double *Ar = A + (size_t)c2 * N; double *Br = B + (size_t)c2 * N;
  for (int c8 = 1; c8 < N; ++c8) Br[c8] -= Ar[c8] * Ar[c8] / Br[c8 - 1];
  for (int c8 = 1; c8 < N; ++c8) Xr[c8] -= Xr[c8 - 1] * Ar[c8] / Br[c8 - 1];
  for (int c8 = 0; c8 <= N - 3; ++c8) Xr[N - 2 - c8] = (Xr[N - 2 - c8] - Xr[N - 3 - c8] * Ar[N - 3 - c8]) / Br[N - 3 - c8];
  Xr[N - 1] /= Br[N - 1];
}
__global__ void row_sweep_k(double *X, const double *A, double *B, int split) {
  int c2 = blockIdx.x * blockDim.x + threadIdx.x;
  if (c2 >= split) return;
  for (int c8 = 1; c8 < N; ++c8) B[(size_t)c8 * N + c2] -= A[(size_t)c8 * N + c2] * A[(size_t)c8 * N + c2] / B[(size_t)(c8 - 1) * N + c2];
  for (int c8 = 1; c8 < N; ++c8) X[(size_t)c8 * N + c2] -= X[(size_t)(c8 - 1) * N + c2] * A[(size_t)c8 * N + c2] / B[(size_t)(c8 - 1) * N + c2];
  for (int c8 = 0; c8 <= N - 3; ++c8) X[(size_t)(N - 2 - c8) * N + c2] = (X[(size_t)(N - 2 - c8) * N + c2] - X[(size_t)(N - 3 - c8) * N + c2] * A[(size_t)(N - 3 - c8) * N + c2]) / B[(size_t)(N - 2 - c8) * N + c2];
  X[(size_t)(N - 1) * N + c2] /= B[(size_t)(N - 1) * N + c2];
}

static void cpu_col(double *X, const double *A, double *B, int r0, int r1) {
#pragma omp parallel for schedule(static)
  for (int c2 = r0; c2 < r1; ++c2) {
    double *Xr = X + (size_t)c2 * N; const double *Ar = A + (size_t)c2 * N; double *Br = B + (size_t)c2 * N;
    for (int c8 = 1; c8 < N; ++c8) Br[c8] -= Ar[c8] * Ar[c8] / Br[c8 - 1];
    for (int c8 = 1; c8 < N; ++c8) Xr[c8] -= Xr[c8 - 1] * Ar[c8] / Br[c8 - 1];
    for (int c8 = 0; c8 <= N - 3; ++c8) Xr[N - 2 - c8] = (Xr[N - 2 - c8] - Xr[N - 3 - c8] * Ar[N - 3 - c8]) / Br[N - 3 - c8];
    Xr[N - 1] /= Br[N - 1];
  }
}
static void cpu_row(double *X, const double *A, double *B, int c0, int c1) {
#pragma omp parallel for schedule(static)
  for (int c2 = c0; c2 < c1; ++c2) {
    for (int c8 = 1; c8 < N; ++c8) B[(size_t)c8 * N + c2] -= A[(size_t)c8 * N + c2] * A[(size_t)c8 * N + c2] / B[(size_t)(c8 - 1) * N + c2];
    for (int c8 = 1; c8 < N; ++c8) X[(size_t)c8 * N + c2] -= X[(size_t)(c8 - 1) * N + c2] * A[(size_t)c8 * N + c2] / B[(size_t)(c8 - 1) * N + c2];
    for (int c8 = 0; c8 <= N - 3; ++c8) X[(size_t)(N - 2 - c8) * N + c2] = (X[(size_t)(N - 2 - c8) * N + c2] - X[(size_t)(N - 3 - c8) * N + c2] * A[(size_t)(N - 3 - c8) * N + c2]) / B[(size_t)(N - 2 - c8) * N + c2];
    X[(size_t)(N - 1) * N + c2] /= B[(size_t)(N - 1) * N + c2];
  }
}

int main(int argc, char **argv) {
  double f = (argc > 1) ? atof(argv[1]) : 0.5;
  int reps = (argc > 2) ? atoi(argv[2]) : 3;
  int split = gpu_split(f, N);
  size_t NN = (size_t)N * N, B2 = NN * sizeof(double);

  double *X, *A, *Bm, *X0, *B0;
  CUDA_CHECK(cudaMallocHost(&X, B2)); CUDA_CHECK(cudaMallocHost(&A, B2)); CUDA_CHECK(cudaMallocHost(&Bm, B2));
  X0 = (double *)malloc(B2); B0 = (double *)malloc(B2);
  for (int i = 0; i < N; ++i) for (int j = 0; j < N; ++j) {
    X0[(size_t)i * N + j] = (double)(i * (j + 1) + 1) / N;
    A[(size_t)i * N + j] = (double)(i * (j + 2) + 2) / N;
    B0[(size_t)i * N + j] = (double)(i * (j + 3) + 3) / N + 1.5;  // +offset keeps the Thomas recurrence numerically stable
  }
  double *dX, *dA, *dB; CUDA_CHECK(cudaMalloc(&dX, B2)); CUDA_CHECK(cudaMalloc(&dA, B2)); CUDA_CHECK(cudaMalloc(&dB, B2));
  CUDA_CHECK(cudaMemcpy(dA, A, B2, cudaMemcpyHostToDevice));
  cudaStream_t s; CUDA_CHECK(cudaStreamCreate(&s));
  int blk = 128, grd = (split + blk - 1) / blk;

  std::vector<double> tot, tcv, tgv;
  for (int r = 0; r < reps + 1; ++r) {
    memcpy(X, X0, B2); memcpy(Bm, B0, B2);
    CUDA_CHECK(cudaMemcpy(dX, X, B2, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, Bm, B2, cudaMemcpyHostToDevice));
    double tc = 0, tg = 0, a, b;
    auto t0 = clk::now();
    for (int t = 0; t <= TSTEPS; ++t) {
      // ---- column sweep: split rows ----
      run_concurrent(split < N, split > 0,
        [&] { if (split < N) cpu_col(X, A, Bm, split, N); },
        [&] { if (split > 0) { col_sweep_k<<<grd, blk, 0, s>>>(dX, dA, dB, split); CUDA_CHECK(cudaStreamSynchronize(s)); } }, a, b);
      tc += a; tg += b;
      if (split > 0) {  // GPU rows [0,split) -> host (contiguous)
        CUDA_CHECK(cudaMemcpy(X, dX, (size_t)split * N * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(Bm, dB, (size_t)split * N * sizeof(double), cudaMemcpyDeviceToHost));
      }
      if (split < N) {  // CPU rows [split,N) -> device
        CUDA_CHECK(cudaMemcpy(dX + (size_t)split * N, X + (size_t)split * N, (size_t)(N - split) * N * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dB + (size_t)split * N, Bm + (size_t)split * N, (size_t)(N - split) * N * sizeof(double), cudaMemcpyHostToDevice));
      }
      // ---- row sweep: split columns ----
      run_concurrent(split < N, split > 0,
        [&] { if (split < N) cpu_row(X, A, Bm, split, N); },
        [&] { if (split > 0) { row_sweep_k<<<grd, blk, 0, s>>>(dX, dA, dB, split); CUDA_CHECK(cudaStreamSynchronize(s)); } }, a, b);
      tc += a; tg += b;
      if (split > 0) {  // GPU cols [0,split) -> host (strided block)
        CUDA_CHECK(cudaMemcpy2D(X, N * sizeof(double), dX, N * sizeof(double), (size_t)split * sizeof(double), N, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy2D(Bm, N * sizeof(double), dB, N * sizeof(double), (size_t)split * sizeof(double), N, cudaMemcpyDeviceToHost));
      }
      if (split < N) {  // CPU cols [split,N) -> device
        CUDA_CHECK(cudaMemcpy2D(dX + split, N * sizeof(double), X + split, N * sizeof(double), (size_t)(N - split) * sizeof(double), N, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy2D(dB + split, N * sizeof(double), Bm + split, N * sizeof(double), (size_t)(N - split) * sizeof(double), N, cudaMemcpyHostToDevice));
      }
    }
    double tt = ms_since(t0);
    if (r > 0) { tot.push_back(tt); tcv.push_back(tc); tgv.push_back(tg); }
  }
  double chk = 0; for (int i = 0; i < N; i += 97) chk += X[(size_t)i * N + (i % N)];
  Result res{median(tot), median(tcv), median(tgv), chk};
  emit_csv("adi", f, res);
  free(X0); free(B0);
  return 0;
}
