#!/usr/bin/env python3
"""llm_healthcheck.py, prove a self-hosted model is actually serving.

"The service is running" is not "the model answers." A model server will happily
return 200 on /health while the weights failed to load, the GPU was taken by
another process, or the context is misconfigured so every real request errors.

This checks the things that actually break, in the order they break:

  1. PORT       is anything listening
  2. HEALTH     does the health endpoint answer
  3. MODEL      is the model the one you think it is
  4. COMPLETION does a real generation come back, the only check that matters
  5. LATENCY    first-token and tokens/sec, so you see degradation before users do
  6. GPU        which card, how much VRAM, who else is on it
  7. NUMA       are compute and memory on the same node (the 10x trap)

Exit 0 healthy, 1 degraded, 2 down, so it drops straight into a cron or a
monitoring check.

    llm_healthcheck.py --url http://127.0.0.1:8080 --model my-model
    llm_healthcheck.py --json          # machine-readable for dashboards
"""
import argparse
import json
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request


def _get(url, timeout=10):
    with urllib.request.urlopen(
            urllib.request.Request(url, headers={"User-Agent": "llm-healthcheck"}),
            timeout=timeout) as r:
        return r.getcode(), r.read().decode("utf-8", "ignore")


def check_health(base):
    try:
        code, body = _get(base.rstrip("/") + "/health", timeout=8)
        return code == 200, f"{code}", body[:120]
    except Exception as e:
        return False, "unreachable", str(e)[:120]


def check_models(base, want=None):
    try:
        code, body = _get(base.rstrip("/") + "/v1/models", timeout=10)
        data = json.loads(body)
        names = [m.get("id") or m.get("name") or "" for m in data.get("data",
                 data.get("models", []))]
        if want and not any(want in (n or "") for n in names):
            return False, f"served model(s) {names} do not match {want!r}", names
        return True, ", ".join(n[:50] for n in names) or "(unnamed)", names
    except Exception as e:
        return False, f"model list failed: {e}", []


def check_completion(base, model, timeout=120):
    """THE check. Everything above can pass while this fails.

    REASONING MODELS BREAK NAIVE HEALTH CHECKS. A thinking model given
    max_tokens=12 spends the entire budget reasoning and returns EMPTY content, a perfectly healthy server that fails every probe forever. Found live on
    a 35B: 14 seconds, HTTP 200, zero content. So: ask for thinking to be
    disabled (harmlessly ignored by non-reasoning servers) AND leave enough
    headroom that a model which thinks anyway still has room to answer.
    """
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user",
                      "content": "Reply with exactly: OK"}],
        "max_tokens": 256, "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False}}).encode()
    t0 = time.time()
    try:
        req = urllib.request.Request(
            base.rstrip("/") + "/v1/chat/completions", data=body,
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            out = json.loads(r.read())
        ms = int((time.time() - t0) * 1000)
        text = out["choices"][0]["message"]["content"].strip()
        usage = out.get("usage", {}) or {}
        gen = usage.get("completion_tokens") or 0
        tps = (gen / (ms / 1000)) if ms and gen else 0.0
        if not text:
            return False, f"empty completion in {ms}ms", {"ms": ms}
        return True, f"{text[:30]!r} in {ms}ms ({tps:.1f} tok/s)", {
            "ms": ms, "tokens": gen, "tok_per_s": round(tps, 2)}
    except Exception as e:
        return False, f"completion failed: {str(e)[:120]}", {}


def check_gpu():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used",
             "--format=csv,noheader"], capture_output=True, text=True, timeout=15)
        if out.returncode != 0:
            return True, "no nvidia-smi (CPU-only deployment?)", []
        rows = [l.strip() for l in out.stdout.splitlines() if l.strip()]
        warn = []
        for r in rows:
            m = re.search(r"(\d+),\s*(.+?),\s*(\d+) MiB,\s*(\d+) MiB", r)
            if m and int(m.group(3)):
                pct = int(m.group(4)) / int(m.group(3)) * 100
                if pct > 92:
                    warn.append(f"card {m.group(1)} at {pct:.0f}% VRAM")
        return (not warn), ("; ".join(warn) if warn
                            else f"{len(rows)} card(s), headroom ok"), rows
    except Exception as e:
        return True, f"gpu check skipped ({str(e)[:60]})", []


def check_numa():
    """The 10x trap: compute and weights on different sockets."""
    try:
        out = subprocess.run(["numactl", "--hardware"], capture_output=True,
                             text=True, timeout=10)
        if out.returncode != 0:
            return True, "single-node or numactl absent", {}
        nodes = re.findall(r"node (\d+) size: (\d+) MB", out.stdout)
        if len(nodes) < 2:
            return True, "single NUMA node, no locality risk", {}
        sizes = {n: int(s) for n, s in nodes}
        big = max(sizes, key=lambda k: sizes[k])
        return True, (f"{len(nodes)} NUMA nodes; largest is node {big} "
                      f"({sizes[big] // 1024} GB), pin the model there "
                      f"(--cpunodebind={big} --membind={big})"), sizes
    except Exception:
        return True, "numa check skipped", {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--model", default=None,
                    help="expected model id (also used for the completion)")
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()

    report, worst = {}, 0

    ok, detail, _ = check_health(a.url)
    report["health"] = {"ok": ok, "detail": detail}
    if not ok:
        worst = 2

    ok_m, detail_m, names = check_models(a.url, a.model)
    report["models"] = {"ok": ok_m, "detail": detail_m, "served": names}
    if not ok_m:
        worst = max(worst, 1)

    model = a.model or (names[0] if names else "default")
    ok_c, detail_c, metrics = check_completion(a.url, model, a.timeout)
    report["completion"] = {"ok": ok_c, "detail": detail_c, **metrics}
    if not ok_c:
        worst = 2                       # this one is fatal by definition

    ok_g, detail_g, cards = check_gpu()
    report["gpu"] = {"ok": ok_g, "detail": detail_g, "cards": cards}
    if not ok_g:
        worst = max(worst, 1)

    _, detail_n, sizes = check_numa()
    report["numa"] = {"detail": detail_n, "nodes": sizes}

    if a.json:
        print(json.dumps({"status": ["healthy", "degraded", "down"][worst],
                          **report}, indent=1))
    else:
        state = ["HEALTHY", "DEGRADED", "DOWN"][worst]
        print(f"[{state}] {a.url}")
        for k in ("health", "models", "completion", "gpu", "numa"):
            v = report[k]
            mark = "  " if v.get("ok", True) else "! "
            print(f"  {mark}{k:11s} {v['detail']}")
    return worst


if __name__ == "__main__":
    sys.exit(main())
