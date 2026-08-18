#!/usr/bin/env python3
"""numa_bench.py - measure what actually limits MoE inference on YOUR box.

WHY THIS EXISTS

Every guide about self-hosting a large model talks about the GPU. For a Mixture-of-Experts
model that is CPU-offloaded, the GPU is rarely the ceiling: the expert weights stream from
system RAM on every token, so the real limit is how fast this machine can read memory, and on
a dual-socket box that number is not one number.

The advice you will find is "use numactl --interleave=all for NUMA machines." On the box this
was written for, interleaving is 23% SLOWER than simply pinning to the better node. That is
not a subtle tuning detail, it is the difference between a usable model and an unusable one,
and you cannot know which applies to you without measuring.

So: measure, do not assume. Run this before you tune anything else.

    numa_bench.py               # full sweep, human readable
    numa_bench.py --json        # machine readable
    numa_bench.py --threads 20  # override thread count per run

Requires numpy and (for the per-node runs) numactl. Degrades gracefully without either.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time

_WORKER = r'''
import sys, time, threading
import numpy as np
threads = int(sys.argv[1]); gib = float(sys.argv[2])
N = int(gib * (1 << 30) / 8)
a = np.ones(N, dtype=np.float64); b = np.empty_like(a)
chunk = N // threads
def w(i):
    lo = i * chunk; hi = N if i == threads - 1 else (i + 1) * chunk
    np.copyto(b[lo:hi], a[lo:hi])
best = 0.0
for _ in range(3):                      # best of 3: we want the ceiling, not the average
    ts = [threading.Thread(target=w, args=(i,)) for i in range(threads)]
    t0 = time.perf_counter()
    for t in ts: t.start()
    for t in ts: t.join()
    dt = time.perf_counter() - t0
    best = max(best, (a.nbytes * 2) / dt / 1e9)   # read + write
print(f"{best:.1f}")
'''


def have(cmd):
    return shutil.which(cmd) is not None


def nodes():
    """Parse numactl --hardware into {node: size_mb}. Empty dict if unavailable."""
    if not have("numactl"):
        return {}
    try:
        out = subprocess.run(["numactl", "--hardware"], capture_output=True,
                             text=True, timeout=15).stdout
    except Exception:
        return {}
    found = {}
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 4 and parts[0] == "node" and parts[2] == "size:":
            found[int(parts[1])] = int(parts[3])
    return found


def run(prefix, threads, gib):
    cmd = list(prefix) + [sys.executable, "-c", _WORKER, str(threads), str(gib)]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
        return float(r.stdout.strip())
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=0, help="per-run thread count")
    ap.add_argument("--gib", type=float, default=2.4, help="buffer size per run")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()

    ns = nodes()
    cpus = os.cpu_count() or 4
    threads = a.threads or max(4, (cpus // max(1, len(ns) or 1)))

    results, notes = {}, []
    if not ns:
        notes.append("numactl unavailable: single-node result only, which is the honest "
                     "answer rather than a guessed topology")
        results["single"] = run([], threads, a.gib)
    else:
        for n, mb in sorted(ns.items()):
            results[f"node{n}-local"] = run(
                ["numactl", f"--cpunodebind={n}", f"--membind={n}"], threads, a.gib)
        # the classic mistake: compute on one socket, memory on the other
        keys = sorted(ns)
        if len(keys) > 1:
            results[f"cpu{keys[0]}-mem{keys[1]}"] = run(
                ["numactl", f"--cpunodebind={keys[0]}", f"--membind={keys[1]}"],
                threads, a.gib)
        results["interleave-all"] = run(["numactl", "--interleave=all"], cpus, a.gib)

    if a.json:
        print(json.dumps({"threads": threads, "cpus": cpus,
                          "nodes": {str(k): v for k, v in ns.items()},
                          "gbs": results, "notes": notes}, indent=2))
        return 0

    print("MEMORY BANDWIDTH (GB/s, read+write, best of 3)\n")
    if ns:
        for n, mb in sorted(ns.items()):
            print(f"  node {n}: {mb/1024:.0f} GiB")
        print()
    width = max(len(k) for k in results) if results else 10
    best = max((v for v in results.values() if v), default=0) or 1
    for k, v in results.items():
        if v is None:
            print(f"  {k:<{width}}  unavailable")
            continue
        bar = "#" * int(38 * v / best)
        print(f"  {k:<{width}}  {v:6.1f}  {bar}")

    local = [v for k, v in results.items() if k.endswith("-local") and v]
    inter = results.get("interleave-all")
    cross = next((v for k, v in results.items() if k.startswith("cpu") and v), None)
    print()
    if local and cross:
        print(f"  local is {max(local)/cross:.1f}x faster than cross-node. Pin compute, "
              f"memory and GPU to ONE node.")
    if local and inter and max(local) > inter:
        print(f"  interleave-all is {(1 - inter/max(local))*100:.0f}% SLOWER than pinning to "
              f"the best node. The common advice does not apply here. Measure yours.")
    if len(local) > 1 and min(local) > 0 and max(local) / min(local) > 1.15:
        print(f"  the nodes differ by {(max(local)/min(local) - 1)*100:.0f}%: the memory "
              f"banks are populated asymmetrically. Put the model on the fast one.")
    for n in notes:
        print(f"  note: {n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
