# local-llm-deploy

Running a large language model on your own hardware, in production, for real users — with
the measured numbers from a deployment that has been serving continuously since June 2026.

This is the guide I wanted when I started: not "install Ollama and you're done," but the
things that actually decide whether a self-hosted model is usable or a toy — memory
bandwidth, NUMA locality, context budgeting, and keeping the thing warm.

**Who needs this:** law firms, medical practices, financial advisors, and anyone else who
cannot send client documents to a public API. Private RAG is not a preference for them, it
is a compliance requirement.

---

## THE FINDING THAT MATTERS MOST

> **For a Mixture-of-Experts model, memory bandwidth decides your speed — not the GPU.**

MoE models activate a fraction of their weights per token (a 1T-parameter model may use
~32B per token). Those weights are streamed from system RAM. So the ceiling is how fast
your CPU can read memory, and the GPU mostly holds attention and the active experts.

| Platform | Channels | Theoretical | Practical effect |
|---|---|---|---|
| Xeon E5 v4, DDR4-2133 | 4 | ~68 GB/s | measured **38 GB/s** — a 1T MoE crawls at 1–2 tok/s |
| EPYC Genoa, DDR5-4800 | 12 | **460 GB/s** | same model, ~18–21 tok/s on published benchmarks |

**Twelve times the bandwidth.** If you are speccing hardware for local inference and you
spend the budget on the GPU instead of the memory platform, you will have bought the wrong
machine. Buy channels first.

---

## NUMA: the mistake that cost me a 10x slowdown

On a dual-socket server, memory is split between sockets. Reading memory attached to the
*other* socket crosses an interconnect and it is dramatically slower.

Our machine:

```
node 0: 20 cores, 96 GB      <- the big bank
node 1: 20 cores, 64 GB
```

The model was pinned to a GPU on node 1 while its expert weights lived in node 0's RAM.
Every single token crossed the interconnect. **Measured result: 0.9 tok/s.** After pinning
compute, memory, and the GPU to the *same* node:

```bash
numactl --cpunodebind=0 --membind=0 llama-server --cpu-moe ...
```

**9.5 tok/s — better than ten times faster, same hardware, one flag.**

The rule: for CPU-offloaded MoE, the GPU, the threads, and the weights must all live on one
node. A 40-thread run spread across both sockets is *slower* than a 20-thread run on one.

If you take one thing from this repo, take this: **check `numactl --hardware` before you
tune anything else.**

---

## Context is VRAM, and slots divide it

```
--ctx-size 131072 --parallel 1 --cache-ram 32768
```

Two things people get wrong:

**1. `--parallel 2` does not double your throughput — it halves your context.** Two slots
split the KV cache, so a 131K window becomes 65K per slot. If your prompt is 92K tokens,
the second slot silently makes every request fail. One slot plus a keepwarm beats two
starved ones for most single-tenant deployments.

**2. Quantization is a memory decision before it is a quality one.** Q8 roughly halves the
weights against FP16; Q4 halves again. On a 12 GB card, that is the difference between the
model fitting and not running at all. Measure quality on *your* task before assuming the
bigger quant is worth it — for extraction and classification work the difference is often
invisible, and for reasoning it is not.

---

## Keep it warm, or pay the cold-start every time

A large prompt prefix (a system prompt, a document, a persona) costs real seconds to
process. If it is identical every call, the KV cache can hold it — but only if something
touches it before it is evicted.

We re-warm the shared prefix every 25 minutes. Measured, from the live log:

```
prompt 92,043 tok | cached 90,207 (98.0%) | 13s
```

**98% cache hit.** Without it, that prefix costs minutes of prefill on every single request.
With it, the model is always seconds away. For any deployment where a long system prompt or
a retrieved document set is reused, this single cron job is the difference between "usable"
and "why is it so slow."

Warm the *exact* prefix your requests use. Warming a different one **evicts** the one you
need — a subtle failure that presents as random slowness.

---

## GPU lanes on a multi-card box

With several GPUs and several services, assign lanes explicitly rather than letting
everything grab card 0:

```
card 0  (small)   embeddings / reranking
card 1  (large)   the LLM, ALONE — a dedicated card means no eviction, no contention
card 2  (large)   image generation, speech-to-text, text-to-speech
```

Set it per-service with `CUDA_VISIBLE_DEVICES` in the unit file, not in a shell profile —
services do not read your `.bashrc`. Before the lanes were explicit, a heavy image job would
evict the model's cache and the next request would take minutes for no visible reason.

---

## A realistic sizing table

| Goal | Minimum honest spec | What you get |
|---|---|---|
| 7–8B dense, chat/extraction | 1× 12 GB GPU, 32 GB RAM | fast, genuinely useful for structured tasks |
| 30–35B MoE, CPU-offloaded | 1× 12 GB GPU, 64+ GB **single-node** RAM | ~12 tok/s on DDR4 — usable, not snappy |
| 30–35B, fully resident | 1× 24 GB GPU (or 2× 12 GB) | several times faster |
| 1T MoE (Kimi-class), Q4 | ~600 GB RAM + a 48–96 GB GPU, **12-channel DDR5** | ~18–21 tok/s |

Bandwidth is the number to buy. Channels before cores, channels before GPU.

---

## Reasoning models break naive health checks

Found while building the tool in this repo, on a live server:

```
completion  empty completion in 14341ms     <- HTTP 200, healthy server, zero output
```

A reasoning model given `max_tokens: 12` spends the **entire budget thinking** and returns
empty content. The server is fine. The probe is wrong. Ship that check and you will get
paged forever, or worse, learn to ignore it.

The fix, which works against reasoning and non-reasoning servers alike:

```python
"max_tokens": 256,                                   # room to think AND answer
"chat_template_kwargs": {"enable_thinking": False}   # ignored by servers that don't think
```

```
completion  'OK' in 532ms (3.8 tok/s)        <- same server, correct probe
```

## The operational half nobody writes about

A self-hosted model is a **service**, and services need service discipline:

- **Health checks that read the body, not the status.** A model server can return `200 OK`
  while serving nothing. Check that a real completion comes back.
- **Restart-loop protection.** A misconfigured service fighting for a port will restart
  forever and look "up" the whole time. We had one restart 891 times before anyone noticed.
- **Watch VRAM, not just RAM.** The failure mode is another process quietly taking the card.
- **Verify by artifact.** "The service is running" is not "the model answers." Curl it.

That discipline is the difference between a demo and something a business can rely on.

---

## What this repo is

The written-down version of a deployment that runs every day: a 35B MoE serving a chat
interface, a nightly document pipeline, an image generator, and speech-to-text on one
dual-socket box with consumer GPUs. Every number above is measured on that machine, not
quoted from a vendor page.

If you are standing up private inference for a firm that cannot use a public API, the
mistakes above are the expensive ones, and they are all avoidable.

MIT licensed.
