# Local Coding LLM on Low-Spec Hardware

A working setup for running a **local, offline coding assistant** on a laptop with a
small GPU and limited RAM — no 24GB+ workstation GPU required. Built and tested on:

| Component | Spec |
|---|---|
| CPU | AMD Ryzen 5 5600H (6 cores / 12 threads) |
| RAM | 6.6 GiB total |
| GPU | NVIDIA GTX 1650, 4 GB VRAM |
| OS | Ubuntu 26.04 |

If your machine is in this ballpark (or better), these steps should work for you.

## Why not a giant context window?

It's tempting to want "the biggest context I can get," but on hardware like this it's
the wrong axis to push on. The KV cache that holds context scales with context length
regardless of model size — a 256k-token context needs tens of GB of cache memory on
its own, far more than this entire machine has. The realistic lever here is **model
size + quantization + a context window sized to what the VRAM actually allows**, not
chasing a huge context number that can't fit anyway.

## What's running here

- **Backend: [llama.cpp](https://github.com/ggml-org/llama.cpp)**, built from source
  with CUDA support. Chosen over Ollama specifically for low-VRAM hardware: it gives
  direct control over GPU-layer offload (`-ngl`) and KV-cache quantization
  (`--cache-type-k/v`), which matters a lot when you're working with single-digit GB
  of VRAM. Ollama wraps the same engine but abstracts those knobs away.
- **Client: [aider](https://aider.chat)**, not a tool-calling agent CLI (e.g. opencode).
  Small quantized models frequently produce malformed tool/function calls, which makes
  agentic tool-calling loops flaky. Aider's diff/whole-file edit formats don't depend
  on function calling, so they hold up much better at this model size.
- **Model: [Qwen3.5-4B](https://huggingface.co/unsloth/Qwen3.5-4B-GGUF)**, `Q4_K_M`
  quant (~2.74 GB). See "Model choice" below for why this beat the obvious
  code-specialized alternative.

## Model choice, with sources

Two candidates were compared before settling on one:

| Model | Verdict | Why |
|---|---|---|
| `Qwen/Qwen2.5-Coder-3B-Instruct` | Fallback | Official code-specialized model (Sept 2024). Official technical report ([arXiv:2409.12186](https://arxiv.org/abs/2409.12186), Table 16): **HumanEval 84.1%, MBPP 73.6%, LiveCodeBench pass@1 10.8%**. |
| **`unsloth/Qwen3.5-4B-GGUF`** | **Chosen** | Real official Alibaba model (released March 2026), dense, ~5B params, ~2.74GB at Q4_K_M. Sourced benchmark: **LiveCodeBench v6 = 55.8%** pass@1 ([llm-stats.com](https://llm-stats.com/models/compare/qwen3-235b-a22b-instruct-2507-vs-qwen3.5-4b)), Intelligence Index reasoning score 27 ([artificialanalysis.ai](https://artificialanalysis.ai/articles/qwen3-5-small-models)). |

**Caveat:** the two LiveCodeBench numbers come from different versions/eval protocols
(official paper vs. third-party aggregator, ~18 months apart), so it's not a strict
apples-to-apples comparison. But the gap (55.8% vs 10.8%) is large enough — and
consistent with the general trend of newer base models absorbing far more code
training data than their predecessors — that Qwen3.5-4B was chosen despite not being
code-specialized. If it underperforms in your own usage, `Qwen2.5-Coder-3B-Instruct`
is a documented fallback (same `download_models.sh` pattern, swap the repo/file name).

Being honest about what a model this size actually delivers: expect a capable
assistant for small, localized edits, boilerplate, and simple functions — not reliable
multi-file refactors or subtle logic. That's the realistic bar for a few-GB model on a
4GB GPU, regardless of which one you pick.

## Setup

```bash
# 1. Build llama.cpp with CUDA support (slow — 15-45+ min on a 6-core laptop CPU)
./scripts/build_llama_cpp.sh

# 2. Download the main coding model (~2.74 GB)
./scripts/download_models.sh

# 3. Start the server (default context: 8192 tokens — see "Tuning context" below)
./scripts/start_server.sh 8192

# 4. In another terminal, point aider at it
cd aider && ../.venv/bin/aider --config .aider.conf.yml
```

No Hugging Face login is required — the model repos used here are public and ungated.

### Tuning context size for your card

`start_server.sh` takes the context size as its one argument. Start at 8192 and push
up (`16384`, `32768`, ...) while watching:

```bash
nvidia-smi    # VRAM usage — stay under ~3.8GB to leave headroom
free -h       # make sure you're not swapping heavily
```

Each run's log lands in `logs/server-ctx<N>.log` so you can compare. Whatever the
largest stable value turns out to be on your hardware, that's your real number — not
a guess.

## Optional: vision / PDF-page input

`llama.cpp`'s multimodal support reads **images**, not PDF text directly — render PDF
pages to images first (e.g. `pdftoppm`) before feeding them in.

Adding a vision tower (`mmproj`) costs the same ~0.67GB regardless of which text model
it's paired with, since the vision encoder size doesn't change. That means pairing it
with the smaller Qwen3.5-2B instead of the 4B leaves dramatically more VRAM for KV
cache:

| Setup | Text weights | + mmproj-F16 | VRAM left for KV cache |
|---|---|---|---|
| Qwen3.5-4B Q4_K_M | 2.74 GB | 3.41 GB | ~0.2-0.4 GB → only ~2-4k ctx |
| Qwen3.5-4B Q3_K_M | 2.29 GB | 2.96 GB | ~0.6-0.8 GB → maybe ~8k ctx |
| **Qwen3.5-2B Q4_K_M** | **1.28 GB** | **1.95 GB** | **~1.6-1.8 GB → ~16-32k ctx** |

This repo ships the 2B+mmproj combination as a **separate profile** on a different
port, so it doesn't compromise the main coding setup:

```bash
WITH_VISION=1 ./scripts/download_models.sh
./scripts/start_vision_server.sh 16384   # serves on :8081, main coder stays on :8080
```

## Project layout

```
local-coding-llm-low-spec-setup/
  README.md                    # this file
  logs/                        # build/download/server/test output (gitignored contents)
  scripts/
    lib_log.sh                 # shared logging helpers
    build_llama_cpp.sh         # clone + CUDA build
    download_models.sh         # fetch GGUF weights (main + optional vision)
    start_server.sh            # main coding profile (Qwen3.5-4B, text-only)
    start_vision_server.sh     # optional vision/PDF profile (Qwen3.5-2B + mmproj)
  aider/
    .aider.conf.yml            # aider config: local endpoint, edit format, theme
  models/                      # gitignored — weights are multi-GB; scripts fetch them
```

## Results observed on the reference hardware

End-to-end tested: built, served, and used through aider to add a function to a
scratch file. Aider produced a valid SEARCH/REPLACE diff on the first try and applied
it cleanly. Numbers below are from that run (`logs/server-ctx8192.log`,
`logs/aider-test-session.log`):

| Metric | Value |
|---|---|
| Context size tested | 8192 tokens, single slot (`--parallel 1`) |
| VRAM used | ~2.9 GB / 4 GB |
| Prompt processing | ~150-190 tokens/sec |
| Generation | ~36-38 tokens/sec |
| Build time (llama.cpp, `BUILD_JOBS=2`) | ~20 min |

One thing worth flagging if you adapt these scripts: llama-server defaults to
splitting `--ctx-size` across 4 parallel request slots, so e.g. `--ctx-size 8192`
would actually give any single conversation only ~2048 tokens unless you also pass
`--parallel 1`. Both server scripts here already set it, since there's only ever one
aider session talking to the server — but it's an easy thing to miss if you tweak the
flags.
