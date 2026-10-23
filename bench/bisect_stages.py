"""Which stage breaks bit identity? Bisect instead of guessing.

This bench exists because one stage did break it, and reasoning about the code
had already produced three wrong answers. Grafting one entry point at a time,
inside a single process on one card in one state, found it in a minute:
`feat_encoder` — and then a second pass over its two input shapes narrowed it
to the prompt-shaped call, which is now excluded by rule.

Keep it. The next person adding a stage will want it, and so will the next
upstream release.

Named `bisect_stages` rather than `bisect`: running a script from `bench/` puts
that directory first on `sys.path`, and a file called `bisect.py` there shadows
the standard library's, which `random` imports, which `tempfile` imports, which
`torch` imports. Every other bench in this folder fails to start.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

PHRASE = "Je ne sais pas, peut-être que c'est plus simple que ça."

COMBINATIONS = [
    ("sampler only", True, ()),
    ("+ base_lm", True, ("base_lm",)),
    ("+ residual_lm", True, ("residual_lm",)),
    ("+ feat_encoder", True, ("feat_encoder",)),
    ("stacks only", False, ("base_lm", "residual_lm", "feat_encoder")),
    ("everything", True, ("base_lm", "residual_lm", "feat_encoder")),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--reference", required=True)
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    sys.path.insert(0, os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))

    import torch
    from voxcpm import VoxCPM

    import faster_voxcpm

    model = VoxCPM.from_pretrained(args.weights, optimize=False)

    def render():
        torch.manual_seed(args.seed)
        start = time.perf_counter()
        wav = model.generate(text=PHRASE, reference_wav_path=args.reference)
        torch.cuda.synchronize()
        return (np.asarray(wav, dtype=np.float32).reshape(-1),
                time.perf_counter() - start)

    render()  # upstream's first generate() in a process differs from the rest
    control, control_took = render()
    print(f"control: {control.size} samples in {control_took:.2f} s\n")
    print(f"{'combination':30} {'s':>6} {'gain':>6}  verdict")

    for name, sampler, stacks in COMBINATIONS:
        graft = faster_voxcpm.accelerate(model, sampler=sampler, stacks=stacks)
        try:
            render()  # warm: every capture lands before the measured render
            wav, took = render()
            same = (wav.size == control.size
                    and bool(np.array_equal(control, wav)))
            if same:
                verdict = "bit-identical"
            elif wav.size != control.size:
                verdict = f"DIFFERENT ({wav.size} vs {control.size} samples)"
            else:
                verdict = (f"DIFFERENT (max delta "
                           f"{np.abs(control - wav).max():.2e})")
            failures = f"  FAILURES {graft.failures}" if graft.failures else ""
            print(f"{name:30} {took:6.2f} {control_took / took:5.2f}x  "
                  f"{verdict}{failures}", flush=True)
        finally:
            graft.undo()

    wav, _ = render()
    print(f"\nafter every undo(): control waveform restored "
          f"{bool(np.array_equal(control, wav))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
