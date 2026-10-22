# faster-voxcpm

[VoxCPM2](https://github.com/OpenBMB/VoxCPM) inference, **2.7x faster**, with a waveform
**identical to the bit**.

No retraining, no quantization, no rewritten model. Four entry points are captured in CUDA
graphs and replayed; a graph changes no kernel, no weight and no shape, only the dispatch.

```python
from voxcpm import VoxCPM
from faster_voxcpm import accelerate

model = VoxCPM.from_pretrained("/path/to/VoxCPM2")
accelerate(model)              # that is the whole integration

wav = model.generate(text="...", reference_wav_path="reference.wav")
```

## Measured

RTX 3070 Ti Laptop (8 GB, Windows / WDDM), bfloat16, seed fixed. Lower RTF is better.

| case | audio | upstream | faster-voxcpm | speedup | waveform |
|---|---|---|---|---|---|
| short, fr | 1.92 s | 4.52 s · RTF 2.352 | 1.51 s · **RTF 0.788** | 2.99x | identical |
| long, fr | 4.48 s | 8.86 s · RTF 1.977 | 3.33 s · **RTF 0.744** | 2.66x | identical |
| english | 2.08 s | 4.18 s · RTF 2.010 | 1.58 s · **RTF 0.759** | 2.65x | identical |
| spanish | 2.08 s | 4.09 s · RTF 1.967 | 1.62 s · **RTF 0.777** | 2.53x | identical |
| style instruction | 4.96 s | 10.19 s · RTF 2.054 | 3.73 s · **RTF 0.752** | 2.73x | identical |

**5/5 bit-identical. Median 2.66x (2.53–2.99x). 0.43–0.51x → 1.27–1.34x real time.
Peak memory 5662 MB → 5701 MB (+40 MB).**

Two runs of the same bench gave medians of 2.66x and 2.70x, so read the figure as "about two
and two thirds", not as three significant digits. The bit-identity column, unlike the clock,
does not vary.

Bit identity is the point, not a bonus. A faster voice engine that does not produce the same
waveform is a different model, so [`bench/verify.py`](bench/verify.py) checks the waveform
before it looks at the clock — and it is what caught the one stage that had to be rolled back.

## Why it is this fast

VoxCPM2 is a MiniCPM4 backbone, a flow-matching DiT head and an audio VAE. One autoregressive
step yields `patch_size: 4` latent frames — 160 ms of audio — and spends it on four transformer
stacks:

| | layers | width | per step |
|---|---|---|---|
| `base_lm` | 28 | 2048 | 1 token |
| flow head (`feat_decoder`) | 12 | 1024 | batch 2 (guidance), **10 times** |
| `feat_encoder` | 12 | 1024 | one patch |
| `residual_lm` | 8 | 2048 | 1 token |

The tensors are tiny — the flow head works on `(2, 64, 4)` — so the arithmetic is nothing.
What gets paid is the **dispatch** of a few thousand kernels per step, and this machine takes
**12.4 microseconds** to launch a trivial kernel where Linux takes 5–10, because Windows runs
the WDDM driver model. Quantization would not touch any of that.

Profiled before anything was written, on 2.88 s of audio:

| component | ms | calls | share |
|---|---|---|---|
| flow head, 10 steps | 3632.7 | 18 | **65%** |
| everything else | 1991.1 | | 35% |

And again once the flow head was graphed, which is how the second stage was aimed:

| component | ms | calls | share |
|---|---|---|---|
| `base_lm.forward_step` | 2236.1 | 32 | 44% |
| flow head (graphed) | 1121.9 | 33 | 22% |
| `feat_encoder` | 794.7 | 34 | 16% |
| `residual_lm.forward_step` | 619.6 | 32 | 12% |
| projections, stop head | ~43 | | ~0% |

A caveat on that second table: the instrumentation synchronises around every call, which on
WDDM is not free — it took the whole run from RTF 0.87 to 0.956. The ranking is sound; the
absolute attribution is generous to the components.

### Why one graph covers a whole utterance

Every shape into the flow sampler is constant across the decode loop. `mu` is one hidden
vector per step, and `cond` is `prefix_feat_cond`, which the loop reassigns to the *latest
patch only* (`prefix_feat_cond = pred_feat`), so it never grows.

The two language models were already decoding against a `StaticKVCache`: a preallocated
buffer written at `key_cache[:, :, position_id, :]` with `position_id` as a tensor, and a mask
built as `arange(...) <= position_id`. Shapes constant, only values change — which is exactly
what a graph wants. Nothing had to be rewritten to make that true.

### What stays outside the capture

- **The noise draw.** It stays in the global RNG stream so a fixed seed still yields the same
  voice; the order, shape and dtype of that draw are what make a seed reproducible.
- **`t_span`**, built by `linspace` and bent by the sway coefficient. It depends only on the
  step count, so it is a constant — built once, outside, and *held*.

## Graph only what repeats

`feat_encoder` is called with two shapes: once per utterance on the whole audio prompt
(60 patches), and once per decode step on a single patch. Fifteen of sixteen calls are the
single-patch one.

Capturing the prompt-shaped call is worth nothing, and it turned out to be worth **less** than
nothing:

| what is graphed | result |
|---|---|
| both shapes | DIFFERENT (99840 vs 115200 samples) |
| **hot shape only** | **identical to the bit** |
| prompt shape only | DIFFERENT (99840 vs 115200 samples) |

**The cause was not found, and that is stated rather than papered over.** Three hypotheses were
tested and refuted: the graph is numerically exact on arbitrary inputs at both shapes, a capture
does not shift the global RNG stream, and neither does a replay. So this is a rule backed by
measurement, not an explanation — and it is the right rule regardless, because a shape seen once
per utterance does not deserve a graph and is exactly where side effects hide.

## The trap, for anyone capturing their own graphs

A CUDA graph records **addresses, not values**. Every tensor it reads must stay alive for as
long as the graph does, *and that includes the constants*.

The sibling project for Ming-3.0-tts learned this the hard way: the flow sampler's timestep
tensor was kept in a local variable of the build function, so it was collected on return, its
device memory went back to the caching allocator, and the next large allocation wrote over it.
The graph then replayed with arbitrary timesteps. The symptom is deceptive in the worst way —
**exact right after capture, wrong later** — and two benches chased the wrong suspects first.

That is why [`faster_voxcpm/capture.py`](faster_voxcpm/capture.py) has a `Captured` dataclass
whose only job is to hold the graph *and* everything it touches.

A second hazard, specific to a stateful cache: capture is not side-effect free. The warmup
passes write keys into the KV cache at whatever position the static buffer holds, and by the
first decode step the prompt already occupies positions 0..n-1. So the cache is snapshotted
before a capture and restored after it — one clone of a few tens of megabytes, once per
process, against an entire class of "the first sentence after a restart is wrong".

## Run the benches yourself

```bash
python bench/verify.py  --weights /path/to/VoxCPM2 --reference /path/to/reference.wav
python bench/profile_step.py --weights ... --reference ...
python bench/profile_step.py --weights ... --reference ... --accelerate
python bench/bisect_stages.py --weights ... --reference ...   # one stage at a time
```

`profile_step.py` starts by timing a trivial kernel launch, because without that floor "a step
costs 70 ms" does not tell you whether you are compute-bound or dispatch-bound — and that is
the whole difference between reaching for quantization and reaching for CUDA graphs.

Tests:

```bash
FASTER_VOXCPM_WEIGHTS=... FASTER_VOXCPM_REFERENCE=... pytest
```

They skip without those variables, so the suite is runnable by someone reading the code who has
no 4.7 GB checkpoint on disk.

## What this does not do

- **It does not change the voice.** Same weights, same output, same quality.
- **It does not shrink the model.** Peak memory is 5.7 GB and the graphs add 40 MB. 81% of the
  weights sit in the two language models (`base_lm` 2807 MB, `residual_lm` 720 MB) against
  404 MB for the flow head, so weight quantization is a separate and complementary lever — and
  one that would want measuring on both axes, since int8 at batch 1 can be slower than bf16.
- **It does not stream.** Upstream has `generate_streaming()`; nothing here touches it, and
  the graphs should apply, but that is untested.
- **Batch size 1**, which is what the decode loop does anyway.
- **The numbers are from one laptop.** The win comes from removing dispatch overhead, so it is
  largest where dispatch is most expensive: Windows/WDDM, laptop GPUs, small batches. On Linux
  with a datacenter card, expect less. Run the bench.

## Requirements

- `voxcpm` installed (developed against 2.0.3) and a VoxCPM2 checkpoint
- `torch` with CUDA — developed on 2.6.0+cu124

## License

MIT. Upstream VoxCPM2 is Apache-2.0.
