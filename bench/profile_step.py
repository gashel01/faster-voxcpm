"""Where do the milliseconds of a decode step go?

THE NUMBER THAT MAKES THE REST READABLE

This bench starts by timing a trivial kernel launch. Without that floor,
"base_lm takes 70 ms for one token" does not say whether the machine is
compute-bound or dispatch-bound, and that is the whole difference between
reaching for quantization and reaching for CUDA graphs. Linux lands around
5-10 microseconds; Windows, under WDDM, 12-18.

ONE HONEST CAVEAT ABOUT THIS BENCH

It synchronises around every wrapped call, which on WDDM is not free: the same
utterance goes from RTF 0.87 unprofiled to 0.956 profiled. Read the ranking,
not the absolute attribution — the components are flattered by roughly the
cost of the barriers around them.

AND ONE LESSON ABOUT WHERE TO ATTACH

The first version of this wrapped `forward` on each submodule and reported
nothing for the backbone, leaving 35% of the turn unattributed. The decode loop
never calls `forward`: it calls `base_lm.forward_step`, because those models
carry their own KV cache with an explicit step. Wrap the call the loop actually
makes.
"""

from __future__ import annotations

import argparse
import os
import statistics as st
import sys
import time
from collections import defaultdict

TEXT = ("Je ne sais pas, peut-être que c'est plus simple que ça. "
        "Enfin, je vais regarder ce soir, quand tout sera calme.")


def launch_floor():
    """Cost of one trivial kernel, launched from Python, on this card."""
    import torch

    x = torch.zeros(1, device="cuda")
    n = 2000
    for _ in range(2):
        for _ in range(n):
            x.add_(1.0)
    torch.cuda.synchronize()
    samples = []
    for _ in range(5):
        start, end = torch.cuda.Event(True), torch.cuda.Event(True)
        start.record()
        for _ in range(n):
            x.add_(1.0)
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end))
    return st.median(samples) / n * 1000


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--reference", required=True)
    ap.add_argument("--accelerate", action="store_true",
                    help="profile the grafted path instead of upstream")
    args = ap.parse_args()

    sys.path.insert(0, os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))

    import numpy as np
    import torch
    from voxcpm import VoxCPM

    print(f"{torch.cuda.get_device_name(0)} - torch {torch.__version__}")
    print(f"launch floor: {launch_floor():.1f} microseconds per trivial "
          f"kernel (Linux is typically 5-10, Windows/WDDM 12-18)")

    model = VoxCPM.from_pretrained(args.weights)
    tts = model.tts_model
    if args.accelerate:
        import faster_voxcpm
        faster_voxcpm.accelerate(model)
        print("grafted")

    times: dict[str, float] = defaultdict(float)
    calls: dict[str, int] = defaultdict(int)

    def wrap(owner, attribute, label):
        """Wrap a call, never a registered submodule.

        `nn.Module.__setattr__` refuses anything but a module over a declared
        child, so for a submodule we wrap its `forward` instead.
        """
        target = getattr(owner, attribute, None)
        if target is None:
            return
        if isinstance(target, torch.nn.Module):
            owner, attribute, target = target, "forward", target.forward
        real = target

        def measured(*a, **k):
            torch.cuda.synchronize()
            start = time.perf_counter()
            out = real(*a, **k)
            torch.cuda.synchronize()
            times[label] += time.perf_counter() - start
            calls[label] += 1
            return out

        setattr(owner, attribute, measured)

    wrap(tts.base_lm, "forward_step", "base_lm.forward_step (28 layers)")
    wrap(tts.residual_lm, "forward_step", "residual_lm.forward_step (8)")
    wrap(tts, "feat_encoder", "feat_encoder (12 layers)")
    wrap(tts, "feat_decoder", "flow head (10 steps)")
    for attribute in ("stop_head", "fsq_layer", "fusion_concat_proj",
                      "lm_to_dit_proj", "res_to_dit_proj", "enc_to_lm_proj"):
        wrap(tts, attribute, attribute)

    model.generate(text="Bonjour.", reference_wav_path=args.reference)
    times.clear()
    calls.clear()

    start = time.perf_counter()
    wav = model.generate(text=TEXT, reference_wav_path=args.reference)
    torch.cuda.synchronize()
    total = time.perf_counter() - start
    seconds = np.asarray(wav, dtype=np.float32).reshape(-1).size / 48000

    print(f"\n{seconds:.2f} s of audio in {total:.2f} s "
          f"({seconds / total:.2f}x real time, RTF {total / seconds:.3f})")
    print(f"\n{'component':34} {'ms':>9} {'calls':>7} {'share':>7} "
          f"{'ms/call':>9}")
    for label in sorted(times, key=lambda n: -times[n]):
        ms = times[label] * 1000
        print(f"{label:34} {ms:9.1f} {calls[label]:7d} "
              f"{100 * times[label] / total:6.0f}% "
              f"{ms / max(calls[label], 1):9.2f}")
    rest = total - sum(times.values())
    print(f"{'unattributed (Python, tqdm…)':34} {rest * 1000:9.1f} "
          f"{'':7} {100 * rest / total:6.0f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
