/*
 * Concurrent CPU+GPU co-execution of a 3x3 stencil (Gaussian-style blur),
 * representative of HeteroBench memory-bound image kernels (sbf / ced).
 *
 * Output rows are partitioned by fraction f exactly as in coexec_gemm.cu:
 * GPU handles rows [0, split), CPU handles rows [split, H), concurrently.
 * Both read the full input image (disjoint output rows => no dependency).
 *
 * This kernel has LOW arithmetic intensity (~9 FMA per 9 loads), so it is
 * MEMORY-BOUND: the optimal split is governed by the bandwidth roofline,
 * not the FLOP roofline.
 *
 * Usage: ./coexec_stencil H W frac reps
 * Output: stencil,H,W,frac,t_total_ms,t_cpu_ms,t_gpu_ms,gbps
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
#include <thread>
#include <chrono>
#include <cuda_runtime.h>
#include <omp.h>

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = (call);                                                  \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,            \
              cudaGetErrorString(err));                                        \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

__constant__ float kGauss[9] = {0.0625f, 0.125f, 0.0625f, 0.125f, 0.25f,
                                0.125f, 0.0625f, 0.125f, 0.0625f};

__global__ void stencil3x3(const float *in, float *out, int rows, int H,
                           int W) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= W || y >= rows) return;
  if (x == 0 || y == 0 || x == W - 1 || y == H - 1) {
    out[y * W + x] = in[y * W + x];
    return;
  }
  float acc = 0.0f;
  int ki = 0;
  for (int dy = -1; dy <= 1; ++dy)
    for (int dx = -1; dx <= 1; ++dx)
      acc += in[(y + dy) * W + (x + dx)] * kGauss[ki++];
  out[y * W + x] = acc;
}

static void cpu_stencil(const float *in, float *out, int row0, int row1, int H,
                        int W) {
  const float k[9] = {0.0625f, 0.125f, 0.0625f, 0.125f, 0.25f,
                      0.125f, 0.0625f, 0.125f, 0.0625f};
#pragma omp parallel for schedule(static)
  for (int y = row0; y < row1; ++y) {
    for (int x = 0; x < W; ++x) {
      if (x == 0 || y == 0 || x == W - 1 || y == H - 1) {
        out[y * W + x] = in[y * W + x];
        continue;
      }
      float acc = 0.0f;
      int ki = 0;
      for (int dy = -1; dy <= 1; ++dy)
        for (int dx = -1; dx <= 1; ++dx)
          acc += in[(y + dy) * W + (x + dx)] * k[ki++];
      out[y * W + x] = acc;
    }
  }
}

int main(int argc, char **argv) {
  int H = (argc > 1) ? atoi(argv[1]) : 4096;
  int W = (argc > 2) ? atoi(argv[2]) : 4096;
  double frac = (argc > 3) ? atof(argv[3]) : 0.5;
  int reps = (argc > 4) ? atoi(argv[4]) : 5;
  if (frac < 0) frac = 0;
  if (frac > 1) frac = 1;
  int split = (int)llround(frac * H);

  size_t bytes = (size_t)H * W * sizeof(float);
  float *in, *out;
  CUDA_CHECK(cudaMallocHost(&in, bytes));
  CUDA_CHECK(cudaMallocHost(&out, bytes));
  for (size_t i = 0; i < (size_t)H * W; ++i) in[i] = (float)(i % 251) * 0.01f;

  float *dIn = nullptr, *dOut = nullptr;
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));
  CUDA_CHECK(cudaMalloc(&dIn, bytes));  // full input needed for halo
  if (split > 0) CUDA_CHECK(cudaMalloc(&dOut, (size_t)split * W * sizeof(float)));

  std::vector<double> totals, cputimes, gputimes;
  for (int r = 0; r < reps + 1; ++r) {
    double t_cpu = 0, t_gpu = 0;
    auto t0 = std::chrono::high_resolution_clock::now();

    std::thread gpu_thread;
    if (split > 0) {
      gpu_thread = std::thread([&]() {
        auto g0 = std::chrono::high_resolution_clock::now();
        // need rows [0, split] plus one halo row => copy min(split+1,H) rows
        int copyRows = (split + 1 < H) ? split + 1 : H;
        CUDA_CHECK(cudaMemcpyAsync(dIn, in, (size_t)copyRows * W * sizeof(float),
                                   cudaMemcpyHostToDevice, stream));
        dim3 block(32, 8);
        dim3 grid((W + block.x - 1) / block.x, (split + block.y - 1) / block.y);
        stencil3x3<<<grid, block, 0, stream>>>(dIn, dOut, split, H, W);
        CUDA_CHECK(cudaMemcpyAsync(out, dOut,
                                   (size_t)split * W * sizeof(float),
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        auto g1 = std::chrono::high_resolution_clock::now();
        t_gpu = std::chrono::duration<double, std::milli>(g1 - g0).count();
      });
    }

    auto c0 = std::chrono::high_resolution_clock::now();
    if (split < H) cpu_stencil(in, out, split, H, H, W);
    auto c1 = std::chrono::high_resolution_clock::now();
    t_cpu = std::chrono::duration<double, std::milli>(c1 - c0).count();

    if (gpu_thread.joinable()) gpu_thread.join();
    auto t1 = std::chrono::high_resolution_clock::now();
    double t_total = std::chrono::duration<double, std::milli>(t1 - t0).count();

    if (r > 0) {
      totals.push_back(t_total);
      cputimes.push_back(t_cpu);
      gputimes.push_back(t_gpu);
    }
  }

  auto median = [](std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v.empty() ? 0.0 : v[v.size() / 2];
  };
  double t_total = median(totals);
  double t_cpu = median(cputimes);
  double t_gpu = median(gputimes);
  // bytes moved ~ read 1 image + write 1 image
  double gbps = (2.0 * bytes) / (t_total / 1000.0) / 1e9;

  printf("stencil,%d,%d,%.4f,%.4f,%.4f,%.4f,%.2f\n", H, W, frac, t_total, t_cpu,
         t_gpu, gbps);

  if (dOut) cudaFree(dOut);
  cudaFree(dIn);
  cudaFreeHost(in);
  cudaFreeHost(out);
  cudaStreamDestroy(stream);
  return 0;
}
