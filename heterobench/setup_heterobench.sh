#!/bin/bash
# Clone the HeteroBench suite and apply our patches:
#   - swap clang++ -> g++ / nvc++ (clang offload is unavailable on TWCC)
#   - enable -O3 -march=native optimization for the CPU baseline (critical:
#     without it the MLP/GEMM CPU runs exceed the wall-clock limit)
#
# Usage:  ./setup_heterobench.sh
# Result: ./HeteroBench/ (multi-GB, git-ignored) ready to profile.

set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"

REPO_URL="${HETEROBENCH_URL:-https://github.com/SFU-HiAccel/HeteroBench.git}"

if [ ! -d HeteroBench ]; then
  echo "=== Cloning HeteroBench ==="
  git clone --depth 1 "$REPO_URL" HeteroBench
else
  echo "=== HeteroBench already present, skipping clone ==="
fi

# Locate the config_json/env_config.json inside the clone (layout: HeteroBench/HeteroBench/...)
CFG="$(find HeteroBench -path '*/config_json/env_config.json' | head -1 || true)"
if [ -z "$CFG" ]; then
  echo "!! could not find config_json/env_config.json in the clone" >&2
  exit 1
fi

echo "=== Applying patched compiler config -> $CFG ==="
cp "$CFG" "${CFG}.orig.bak"
cp "$HERE/patches/env_config.json" "$CFG"

echo "=== Done. HeteroBench is ready at: $HERE/HeteroBench ==="
echo "Next: profile with the suite's heterobench.py (run under SLURM on a GPU node)."
echo "Pure-CPU/GPU reference results are in: $HERE/results/heterobench_pure.json"
