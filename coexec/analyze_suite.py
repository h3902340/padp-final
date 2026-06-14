#!/usr/bin/env python3
"""
Summarise the all-benchmark co-execution sweep.

For each HeteroBench benchmark we report the makespan at:
  pure CPU (f=0), pure GPU (f=1), and the best co-execution mix (min over f),
plus the optimal split f* and the speedup of the mix over each single device.
"""
import csv, json
from pathlib import Path

HERE = Path(__file__).parent
CLASS = {
    "3mm": "compute-bound (GEMM)", "mlp": "compute-bound (GEMM)",
    "oha": "compute-bound (attention)", "cnn": "compute/memory (conv+FC)",
    "adi": "memory-bound (stencil sweep)", "ced": "memory-bound (image stencil)",
    "sbf": "memory-bound (image stencil)", "opf": "memory-bound (4K pipeline)",
    "spf": "irregular (sequential SGD)", "ppc": "irregular (n-body)",
}
ORDER = ["3mm", "mlp", "oha", "cnn", "adi", "ced", "sbf", "opf", "spf", "ppc"]


def main():
    csv_path = HERE / "results" / "suite_sweep.csv"
    rows = {}
    with open(csv_path) as fh:
        for r in csv.DictReader(fh):
            try:
                name = r["name"]; f = float(r["frac"]); t = float(r["t_total_ms"])
            except (ValueError, KeyError):
                continue
            rows.setdefault(name, []).append((f, t, float(r["t_cpu_ms"]), float(r["t_gpu_ms"])))

    out = {"description": "HeteroBench concurrent CPU+GPU co-execution: makespan (ms) "
           "by split fraction f. 4 CPU cores + Tesla V100, single precision.",
           "benchmarks": {}}
    for name in ORDER:
        if name not in rows:
            continue
        sweep = sorted(rows[name])
        bypts = {round(f, 3): t for f, t, _, _ in sweep}
        cpu = bypts.get(0.0)
        gpu = bypts.get(1.0)
        f_best, t_best = min(((f, t) for f, t, _, _ in sweep), key=lambda x: x[1])
        d = {
            "class": CLASS.get(name, "?"),
            "pure_cpu_ms": cpu, "pure_gpu_ms": gpu,
            "mix_ms": t_best, "f_star": f_best,
            "speedup_vs_cpu": (cpu / t_best) if cpu else None,
            "speedup_vs_gpu": (gpu / t_best) if gpu else None,
            "best_device": "CPU" if (cpu and gpu and cpu < gpu) else "GPU",
            "mix_wins": bool(cpu and gpu and t_best < min(cpu, gpu) - 1e-6),
            "sweep": [(f, t) for f, t, _, _ in sweep],
        }
        out["benchmarks"][name] = d

    (HERE / "results").mkdir(exist_ok=True)
    with open(HERE / "results" / "suite_results.json", "w") as fh:
        json.dump(out, fh, indent=2)

    print(f"\n{'bench':5} {'class':28} {'CPU(ms)':>9} {'GPU(ms)':>9} "
          f"{'mix(ms)':>9} {'f*':>4} {'vsCPU':>6} {'vsGPU':>6} {'win'}")
    for name in ORDER:
        if name not in out["benchmarks"]:
            continue
        d = out["benchmarks"][name]
        print(f"{name:5} {d['class']:28} {d['pure_cpu_ms']:9.1f} {d['pure_gpu_ms']:9.1f} "
              f"{d['mix_ms']:9.1f} {d['f_star']:4.1f} "
              f"{(d['speedup_vs_cpu'] or 0):6.2f} {(d['speedup_vs_gpu'] or 0):6.2f} "
              f"{'MIX' if d['mix_wins'] else d['best_device']}")


if __name__ == "__main__":
    main()
