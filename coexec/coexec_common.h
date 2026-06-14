/*
 * Shared helpers for the HeteroBench concurrent CPU+GPU co-execution drivers.
 *
 * Co-execution model (identical for every benchmark):
 *   A data-parallel stage with output domain of size M is split by a fraction f:
 *     - GPU computes the first  split = round(f*M) rows/elements  (CUDA, async stream)
 *     - CPU computes the remaining  M - split  rows/elements      (OpenMP, all cores)
 *   The two run AT THE SAME TIME: a host std::thread issues the async GPU work and
 *   blocks on the stream, while the main thread drives the CPU OpenMP loop. The stage
 *   makespan is max(t_cpu, t_gpu). Multi-stage pipelines apply the split per stage
 *   with an implicit barrier (thread join) between stages.
 *
 *   f = 0  -> pure CPU (OpenMP baseline)
 *   f = 1  -> pure GPU
 *   0<f<1  -> concurrent mix
 *
 * All buffers are single precision (float) for memory/throughput uniformity; the
 * kernel MATH and problem SIZES match the HeteroBench reference implementations.
 * Timing is wall-clock milliseconds (median of `reps`, one warm-up excluded),
 * matching HeteroBench's per-iteration millisecond convention.
 */
#ifndef COEXEC_COMMON_H
#define COEXEC_COMMON_H

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#include <thread>
#include <chrono>
#include <functional>
#include <cuda_runtime.h>
#include <omp.h>

#define CUDA_CHECK(call)                                                        \
  do {                                                                          \
    cudaError_t err = (call);                                                   \
    if (err != cudaSuccess) {                                                   \
      fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,             \
              cudaGetErrorString(err));                                         \
      exit(1);                                                                  \
    }                                                                           \
  } while (0)

using clk = std::chrono::high_resolution_clock;
static inline double ms_since(const clk::time_point &t0) {
  return std::chrono::duration<double, std::milli>(clk::now() - t0).count();
}

// Run a CPU lambda and a GPU lambda concurrently; report each side's busy time.
// The GPU lambda must be self-contained (enqueue async work on its stream and
// synchronize before returning).
template <class CpuFn, class GpuFn>
static inline void run_concurrent(bool do_cpu, bool do_gpu, CpuFn cpu,
                                  GpuFn gpu, double &t_cpu, double &t_gpu) {
  t_cpu = 0;
  t_gpu = 0;
  std::thread gt;
  if (do_gpu) {
    gt = std::thread([&]() {
      auto g0 = clk::now();
      gpu();
      t_gpu = ms_since(g0);
    });
  }
  if (do_cpu) {
    auto c0 = clk::now();
    cpu();
    t_cpu = ms_since(c0);
  }
  if (gt.joinable()) gt.join();
}

static inline double median(std::vector<double> v) {
  if (v.empty()) return 0.0;
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

// split = number of rows assigned to the GPU (the first `split` rows).
static inline int gpu_split(double f, int M) {
  if (f < 0) f = 0;
  if (f > 1) f = 1;
  int s = (int)llround(f * M);
  if (s < 0) s = 0;
  if (s > M) s = M;
  return s;
}

// Standard CSV record emitted by every driver.
struct Result {
  double t_total_ms, t_cpu_ms, t_gpu_ms, checksum;
};

static inline void emit_csv(const char *name, double f, const Result &r) {
  printf("%s,%.4f,%.4f,%.4f,%.4f,%.6g\n", name, f, r.t_total_ms, r.t_cpu_ms,
         r.t_gpu_ms, r.checksum);
  fflush(stdout);
}

#endif  // COEXEC_COMMON_H
