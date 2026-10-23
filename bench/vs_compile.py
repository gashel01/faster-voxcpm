"""faster-voxcpm against upstream's own `optimize=True`, which is the fair bar.

WHY THIS BENCH EXISTS

`VoxCPM.from_pretrained` defaults to `optimize=True`, which wraps the same four
entry points this package grafts in `torch.compile(mode="reduce-overhead")` --
CUDA graphs, plus Inductor's kernel fusion. It needs Triton. Without it,
upstream prints a warning and silently runs eager, and that eager path is what
`verify.py` compares against. So `verify.py` answers "faster than what most
Windows installs actually run", and this bench answers the harder question:
faster than upstream at its best?

ONE MODE PER PROCESS

    eager     optimize=False                      upstream, no compilation
    compiled  optimize=True  (needs Triton)       upstream at its best
    faster    optimize=False, then accelerate()   this package

Each mode runs in its own process because compilation state is global and the
allocator never hands its reserve back, so two modes in one process would
contaminate each other's time and memory. Each run saves its waveforms; the
`compare` step then checks every mode against eager, to the bit, and prints
the table.

    python bench/vs_compile.py run --mode eager    --weights W --reference R --out D
    python bench/vs_compile.py run --mode compiled --weights W --reference R --out D
    python bench/vs_compile.py run --mode faster   --weights W --reference R --out D
    python bench/vs_compile.py compare --out D

`--repeats` renders every case that many times and keeps the median, because a
single pass on a laptop moves by 10-20% between runs.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from verify import CASES  # noqa: E402  same sentences as the identity bench

MODES = ("eager", "compiled", "faster")


def run(args) -> int:
    sys.path.insert(0, os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    import torch
    from voxcpm import VoxCPM

    os.makedirs(args.out, exist_ok=True)
    if args.mode == "compiled":
        try:
            import triton  # noqa: F401
        except ImportError:
            print("compiled mode needs Triton; upstream would silently run "
                  "eager and the comparison would mean nothing")
            return 2

    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    model = VoxCPM.from_pretrained(args.weights,
                                   optimize=(args.mode == "compiled"))
    graft = None
    if args.mode == "faster":
        import faster_voxcpm
        graft = faster_voxcpm.accelerate(model)
    rate = int(getattr(model.tts_model, "sample_rate", 48000))

    def render(text):
        torch.manual_seed(args.seed)
        t0 = time.perf_counter()
        wav = model.generate(text=text, reference_wav_path=args.reference)
        torch.cuda.synchronize()
        return (np.asarray(wav, dtype=np.float32).reshape(-1),
                time.perf_counter() - t0)

    # Warm every shape the measured renders will see: compilation and capture
    # both happen on first use, and neither belongs inside a timed render.
    for _, text in CASES:
        render(text)
    ready = time.perf_counter() - start

    result = {"mode": args.mode, "ready_s": ready, "rate": rate,
              "device": torch.cuda.get_device_name(0),
              "torch": torch.__version__, "cases": {}}
    waves = {}
    for name, text in CASES:
        times = []
        for _ in range(args.repeats):
            wav, took = render(text)
            times.append(took)
        waves[name] = wav
        seconds = wav.size / rate
        took = statistics.median(times)
        result["cases"][name] = {"audio_s": seconds, "median_s": took,
                                 "times": times}
        print(f"  {args.mode:8} {name:18} {seconds:5.2f} s audio  "
              f"{took:6.2f} s  ({seconds / took:.2f}x real time)", flush=True)
    result["peak_mb"] = torch.cuda.max_memory_allocated() / 2 ** 20
    if graft is not None:
        result["fallbacks"] = graft.fallbacks
        result["failures"] = graft.failures

    np.savez(os.path.join(args.out, f"{args.mode}.npz"), **waves)
    with open(os.path.join(args.out, f"{args.mode}.json"), "w",
              encoding="utf-8") as f:
        json.dump(result, f, indent=1)
    print(f"  ready after {ready:.1f} s (load, compile or capture, warmup); "
          f"peak {result['peak_mb']:,.0f} MB")
    return 0


def compare(args) -> int:
    runs = {}
    for mode in MODES:
        path = os.path.join(args.out, f"{mode}.json")
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                runs[mode] = json.load(f)
            runs[mode]["waves"] = np.load(os.path.join(args.out,
                                                       f"{mode}.npz"))
    if "eager" not in runs:
        print("run the eager mode first: everything is compared against it")
        return 2

    eager = runs["eager"]
    print(f"{eager['device']}, torch {eager['torch']}\n")
    header = f"  {'case':18} {'audio':>6}"
    for mode in runs:
        header += f"  {mode:>16}"
    print(header)
    for name, _ in CASES:
        line = f"  {name:18} {eager['cases'][name]['audio_s']:5.2f}s"
        ref = eager["waves"][name]
        for mode, r in runs.items():
            c = r["cases"][name]
            wav = r["waves"][name]
            same = ref.size == wav.size and bool(np.array_equal(ref, wav))
            mark = "=" if same else "~"
            line += f"  {c['audio_s'] / c['median_s']:5.2f}x {mark:1} " \
                    f"{eager['cases'][name]['median_s'] / c['median_s']:5.2f}x"
        print(line)
    print("\n  each cell: real-time factor, = bit-identical to eager / "
          "~ different, speedup over eager")
    for mode, r in runs.items():
        speeds = [eager["cases"][n]["median_s"] / r["cases"][n]["median_s"]
                  for n, _ in CASES]
        print(f"  {mode:8} median {statistics.median(speeds):.2f}x over eager, "
              f"ready after {r['ready_s']:.1f} s, peak {r['peak_mb']:,.0f} MB")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--mode", choices=MODES, required=True)
    r.add_argument("--weights", required=True)
    r.add_argument("--reference", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--seed", type=int, default=123)
    r.add_argument("--repeats", type=int, default=3)
    c = sub.add_parser("compare")
    c.add_argument("--out", required=True)
    args = ap.parse_args()
    return run(args) if args.cmd == "run" else compare(args)


if __name__ == "__main__":
    raise SystemExit(main())
