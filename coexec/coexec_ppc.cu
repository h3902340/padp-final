/*
 * Parallelize-particle: grid-binned 2D short-range n-body.
 * Sizes (HeteroBench): NPARTICLES=1,000,000, NSTEPS=5, cutoff=0.01.
 * Co-execution splits the PARTICLE index: GPU does particles [0,split), CPU does
 * [split,N) for both the force computation and the move; the uniform grid is
 * rebuilt on the host each step and mirrored to the device. Irregular gather +
 * divergent cutoff branches make this a CPU-favored workload.
 *
 * Usage: ./coexec_ppc <frac> <reps>  -> CSV: ppc,frac,t_total,t_cpu,t_gpu,checksum
 */
#include "coexec_common.h"

static const int NP = 1000000, NSTEPS = 5, CAP = 8;
#define CUTOFF 0.01
#define MINR (CUTOFF / 100.0)
#define MASS 0.01
#define DT 0.0005

struct Particle { double x, y, vx, vy, ax, ay; };

__device__ __host__ static inline void apply(double px, double py, double qx,
                                             double qy, double &ax, double &ay) {
  double dx = qx - px, dy = qy - py, r2 = dx * dx + dy * dy;
  if (r2 > CUTOFF * CUTOFF) return;
  r2 = fmax(r2, MINR * MINR);
  double r = sqrt(r2), coef = (1 - CUTOFF / r) / r2 / MASS;
  ax += coef * dx; ay += coef * dy;
}

__global__ void forces_k(Particle *P, int split, const int *grid, const int *gc,
                         int GRID) {
  int p = blockIdx.x * blockDim.x + threadIdx.x;
  if (p >= split) return;
  double px = P[p].x, py = P[p].y, ax = 0, ay = 0;
  int cx = (int)(px / CUTOFF), cy = (int)(py / CUTOFF);
  cx = cx < 0 ? 0 : (cx >= GRID ? GRID - 1 : cx);
  cy = cy < 0 ? 0 : (cy >= GRID ? GRID - 1 : cy);
  for (int dx = -1; dx <= 1; ++dx)
    for (int dy = -1; dy <= 1; ++dy) {
      int nx = cx + dx, ny = cy + dy;
      if (nx < 0 || ny < 0 || nx >= GRID || ny >= GRID) continue;
      int cell = nx * GRID + ny, n = gc[cell];
      for (int e = 0; e < n; ++e) { int q = grid[cell * CAP + e]; if (q != p) apply(px, py, P[q].x, P[q].y, ax, ay); }
    }
  P[p].ax = ax; P[p].ay = ay;
}
__global__ void move_k(Particle *P, int split, double SIZE) {
  int p = blockIdx.x * blockDim.x + threadIdx.x;
  if (p >= split) return;
  P[p].vx += P[p].ax * DT; P[p].vy += P[p].ay * DT;
  P[p].x += P[p].vx * DT; P[p].y += P[p].vy * DT;
  while (P[p].x < 0 || P[p].x > SIZE) { P[p].x = P[p].x < 0 ? -P[p].x : 2 * SIZE - P[p].x; P[p].vx = -P[p].vx; }
  while (P[p].y < 0 || P[p].y > SIZE) { P[p].y = P[p].y < 0 ? -P[p].y : 2 * SIZE - P[p].y; P[p].vy = -P[p].vy; }
}

static void cpu_forces(Particle *P, int p0, int p1, const int *grid, const int *gc, int GRID) {
#pragma omp parallel for schedule(dynamic, 4096)
  for (int p = p0; p < p1; ++p) {
    double px = P[p].x, py = P[p].y, ax = 0, ay = 0;
    int cx = (int)(px / CUTOFF), cy = (int)(py / CUTOFF);
    cx = cx < 0 ? 0 : (cx >= GRID ? GRID - 1 : cx);
    cy = cy < 0 ? 0 : (cy >= GRID ? GRID - 1 : cy);
    for (int dx = -1; dx <= 1; ++dx)
      for (int dy = -1; dy <= 1; ++dy) {
        int nx = cx + dx, ny = cy + dy;
        if (nx < 0 || ny < 0 || nx >= GRID || ny >= GRID) continue;
        int cell = nx * GRID + ny, n = gc[cell];
        for (int e = 0; e < n; ++e) { int q = grid[cell * CAP + e]; if (q != p) apply(px, py, P[q].x, P[q].y, ax, ay); }
      }
    P[p].ax = ax; P[p].ay = ay;
  }
}
static void cpu_move(Particle *P, int p0, int p1, double SIZE) {
#pragma omp parallel for schedule(static)
  for (int p = p0; p < p1; ++p) {
    P[p].vx += P[p].ax * DT; P[p].vy += P[p].ay * DT;
    P[p].x += P[p].vx * DT; P[p].y += P[p].vy * DT;
    while (P[p].x < 0 || P[p].x > SIZE) { P[p].x = P[p].x < 0 ? -P[p].x : 2 * SIZE - P[p].x; P[p].vx = -P[p].vx; }
    while (P[p].y < 0 || P[p].y > SIZE) { P[p].y = P[p].y < 0 ? -P[p].y : 2 * SIZE - P[p].y; P[p].vy = -P[p].vy; }
  }
}

int main(int argc, char **argv) {
  double f = (argc > 1) ? atof(argv[1]) : 0.5;
  int reps = (argc > 2) ? atoi(argv[2]) : 3;
  double SIZE = sqrt(0.0005 * NP);
  int GRID = (int)(SIZE / CUTOFF) + 1;
  size_t NCELL = (size_t)GRID * GRID;
  int split = gpu_split(f, NP);

  Particle *P, *P0; int *grid, *gc;
  CUDA_CHECK(cudaMallocHost(&P, NP * sizeof(Particle)));
  P0 = (Particle *)malloc(NP * sizeof(Particle));
  grid = (int *)malloc(NCELL * CAP * sizeof(int));
  gc = (int *)malloc(NCELL * sizeof(int));
  srand(1234);
  for (int i = 0; i < NP; ++i) {
    P0[i].x = (double)rand() / RAND_MAX * SIZE; P0[i].y = (double)rand() / RAND_MAX * SIZE;
    P0[i].vx = ((double)rand() / RAND_MAX - 0.5); P0[i].vy = ((double)rand() / RAND_MAX - 0.5);
    P0[i].ax = P0[i].ay = 0;
  }
  Particle *dP; int *dGrid, *dGc;
  CUDA_CHECK(cudaMalloc(&dP, NP * sizeof(Particle)));
  CUDA_CHECK(cudaMalloc(&dGrid, NCELL * CAP * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&dGc, NCELL * sizeof(int)));
  cudaStream_t s; CUDA_CHECK(cudaStreamCreate(&s));
  int blk = 256, grd = (split + blk - 1) / blk;

  auto build = [&]() {
    memset(gc, 0, NCELL * sizeof(int));
    for (int i = 0; i < NP; ++i) {
      int cx = (int)(P[i].x / CUTOFF), cy = (int)(P[i].y / CUTOFF);
      cx = cx < 0 ? 0 : (cx >= GRID ? GRID - 1 : cx);
      cy = cy < 0 ? 0 : (cy >= GRID ? GRID - 1 : cy);
      int cell = cx * GRID + cy;
      if (gc[cell] < CAP) grid[cell * CAP + gc[cell]++] = i;
    }
  };

  std::vector<double> tot, tcv, tgv;
  for (int r = 0; r < reps + 1; ++r) {
    memcpy(P, P0, NP * sizeof(Particle));
    double tc = 0, tg = 0, a, b;
    auto t0 = clk::now();
    for (int step = 0; step < NSTEPS; ++step) {
      build();
      CUDA_CHECK(cudaMemcpy(dP, P, NP * sizeof(Particle), cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(dGrid, grid, NCELL * CAP * sizeof(int), cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(dGc, gc, NCELL * sizeof(int), cudaMemcpyHostToDevice));
      // forces
      run_concurrent(split < NP, split > 0,
        [&] { if (split < NP) cpu_forces(P, split, NP, grid, gc, GRID); },
        [&] { if (split > 0) { forces_k<<<grd, blk, 0, s>>>(dP, split, dGrid, dGc, GRID); CUDA_CHECK(cudaStreamSynchronize(s)); } }, a, b);
      tc += a; tg += b;
      // move
      run_concurrent(split < NP, split > 0,
        [&] { if (split < NP) cpu_move(P, split, NP, SIZE); },
        [&] { if (split > 0) { move_k<<<grd, blk, 0, s>>>(dP, split, SIZE); CUDA_CHECK(cudaStreamSynchronize(s)); } }, a, b);
      tc += a; tg += b;
      // merge particle positions for next step's grid build
      if (split > 0) CUDA_CHECK(cudaMemcpy(P, dP, (size_t)split * sizeof(Particle), cudaMemcpyDeviceToHost));
    }
    double tt = ms_since(t0);
    if (r > 0) { tot.push_back(tt); tcv.push_back(tc); tgv.push_back(tg); }
  }
  double chk = 0; for (int i = 0; i < NP; i += 99991) chk += P[i].x + P[i].y;
  Result res{median(tot), median(tcv), median(tgv), chk};
  emit_csv("ppc", f, res);
  free(P0); free(grid); free(gc);
  return 0;
}
