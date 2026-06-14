#!/usr/bin/env python3
"""
Progress monitor for HeteroBench AI experiments.

Usage:
  python3 progress.py           # one-shot status
  python3 progress.py --watch   # refresh every 10s
  python3 progress.py -w 5      # refresh every 5s
"""

from __future__ import print_function

import argparse
import glob
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(EXP_DIR, "results")
LOGS_DIR = os.path.join(EXP_DIR, "logs")

BENCHMARKS = [
    ("ced", "Canny Edge Detection"),
    ("sbf", "Sobel Filter"),
    ("3mm", "3 Matrix Multiplication"),
    ("adi", "Alternating Direction Implicit"),
    ("ppc", "Parallelize Particle"),
    ("mlp", "Multilayer Perceptron"),
    ("dgr", "Digit Recognition"),
    ("spf", "Spam Filter"),
    ("cnn", "Convolutional Neural Network"),
    ("oha", "One-Head Attention"),
    ("opf", "Optical Flow"),
]

PHASES = ["cpu", "gpu_omp", "hetero"]

# Total work units: 11 benchmarks × 3 phases + 3 pipeline steps
PIPELINE_STEPS = ["parse", "optimize", "figures"]
TOTAL_UNITS = len(BENCHMARKS) * len(PHASES) + len(PIPELINE_STEPS)


def clear_screen():
    sys.stdout.write("\033[2J\033[H")
    sys.stdout.flush()


def bar(done, total, width=40):
    if total <= 0:
        return "[" + ("=" * width) + "] 100%"
    frac = min(float(done) / total, 1.0)
    filled = int(width * frac)
    empty = width - filled
    pct = int(frac * 100)
    return "[{0}{1}] {2}%".format("=" * filled, "-" * empty, pct)


def log_has_timing(path):
    """Return True if log contains a completed timing result."""
    if not os.path.isfile(path) or os.path.getsize(path) == 0:
        return False
    try:
        with open(path, "r", errors="replace") as f:
            text = f.read()
        return "Single iteration time:" in text or "Total" in text and "time:" in text
    except IOError:
        return False


def parse_timing(path):
    m = re.search(r"Single iteration time:\s+([\d.]+)\s*ms", open(path, errors="replace").read())
    return float(m.group(1)) if m else None


def phase_status(bench, phase):
    log = os.path.join(RESULTS_DIR, "{0}_{1}.log".format(bench, phase))
    if log_has_timing(log):
        t = parse_timing(log)
        return "done", t
    if os.path.isfile(log) and os.path.getsize(log) > 0:
        return "running", None
    return "pending", None


def pipeline_status():
    status = {}
    profile = os.path.join(RESULTS_DIR, "profile_data.json")
    assign = os.path.join(RESULTS_DIR, "optimal_assignments.json")
    figures = os.path.join(EXP_DIR, "figures", "speedup_summary.pdf")

    if os.path.isfile(profile):
        try:
            with open(profile) as f:
                n = len(json.load(f))
            status["parse"] = "done ({0} benchmarks)".format(n)
        except (IOError, ValueError):
            status["parse"] = "done"
    else:
        status["parse"] = "pending"

    if os.path.isfile(assign):
        try:
            with open(assign) as f:
                n = len(json.load(f))
            status["optimize"] = "done ({0} assignments)".format(n)
        except (IOError, ValueError):
            status["optimize"] = "done"
    else:
        status["optimize"] = "pending"

    status["figures"] = "done" if os.path.isfile(figures) else "pending"
    return status


def get_slurm_jobs():
    try:
        out = subprocess.check_output(
            ["squeue", "-u", os.environ.get("USER", "")],
            stderr=subprocess.DEVNULL,
            universal_newlines=True,
        )
    except (subprocess.CalledProcessError, OSError):
        return []
    jobs = []
    for line in out.strip().splitlines()[1:]:
        parts = line.split()
        if len(parts) >= 5:
            jobs.append({"id": parts[0], "name": parts[2], "state": parts[4], "time": parts[5] if len(parts) > 5 else "?"})
    return jobs


def detect_current_benchmark():
    """Guess current benchmark from most recently modified slurm log."""
    logs = glob.glob(os.path.join(LOGS_DIR, "*.out"))
    if not logs:
        return None, None
    latest = max(logs, key=os.path.getmtime)
    try:
        with open(latest, errors="replace") as f:
            text = f.read()
        matches = re.findall(r"--- BUILD/RUN (\w+) (\w+) ---", text)
        if matches:
            return matches[-1][0], matches[-1][1]
        m = re.search(r"BUILD/RUN (\w+) (\w+)", text)
        if m:
            return m.group(1), m.group(2)
    except IOError:
        pass
    return None, None


def count_completed():
    done = 0
    for bench, _ in BENCHMARKS:
        for phase in PHASES:
            st, _ = phase_status(bench, phase)
            if st == "done":
                done += 1
    pipe = pipeline_status()
    for step in PIPELINE_STEPS:
        if pipe[step].startswith("done"):
            done += 1
    return done


def status_icon(st):
    return {"done": "✓", "running": "…", "pending": "·"}.get(st, "?")


def render(watch=False):
    if watch:
        clear_screen()

    done = count_completed()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print("=" * 60)
    print("  HeteroBench Experiment Progress")
    print("  {0}".format(now))
    print("=" * 60)
    print()
    print("  Overall  {0}  ({1}/{2} steps)".format(bar(done, TOTAL_UNITS, 50), done, TOTAL_UNITS))
    print()

    # Per-benchmark breakdown
    print("  Benchmark          CPU    GPU    Hetero   Best time")
    print("  " + "-" * 54)
    for bench, name in BENCHMARKS:
        cols = []
        best = None
        for phase in PHASES:
            st, t = phase_status(bench, phase)
            cols.append("{0} {1}".format(status_icon(st), phase[:3]))
            if t is not None and (best is None or t < best):
                best = t
        best_str = "{0:.1f} ms".format(best) if best else "-"
        print("  {0:<18} {1}  {2}  {3}   {4}".format(
            bench, cols[0], cols[1], cols[2], best_str))

    print()
    print("  Pipeline")
    pipe = pipeline_status()
    for step in PIPELINE_STEPS:
        print("    [{0}] {1}: {2}".format(
            "✓" if pipe[step].startswith("done") else "·",
            step, pipe[step]))

    print()
    jobs = get_slurm_jobs()
    if jobs:
        print("  Active SLURM jobs:")
        for j in jobs:
            print("    {id}  {name:<12}  {state}  {time}".format(**j))
        cur_bench, cur_phase = detect_current_benchmark()
        if cur_bench:
            print("  Currently running: {0} ({1})".format(cur_bench, cur_phase))
    else:
        print("  No active SLURM jobs.")

    print()
    print("  Logs:  {0}".format(LOGS_DIR))
    print("  Results: {0}".format(RESULTS_DIR))
    if not watch:
        print()
        print("  Tip: run with --watch for live updates")


def main():
    parser = argparse.ArgumentParser(description="Monitor HeteroBench experiment progress")
    parser.add_argument("-w", "--watch", type=float, nargs="?", const=10, metavar="SEC",
                        help="Refresh every SEC seconds (default: 10)")
    args = parser.parse_args()

    if args.watch is not None:
        interval = args.watch
        try:
            while True:
                render(watch=True)
                print()
                print("  Refreshing every {0}s — Ctrl+C to stop".format(interval))
                time.sleep(interval)
        except KeyboardInterrupt:
            print("\nStopped.")
    else:
        render(watch=False)


if __name__ == "__main__":
    main()
