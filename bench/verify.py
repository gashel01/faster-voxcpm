"""Does it still say exactly the same thing, and how much faster?

THE ORDER OF THESE TWO QUESTIONS IS NOT NEGOTIABLE

A faster voice engine that does not produce the same waveform is a different
model, not an optimisation. So identity is checked first, to the bit, and only
then the clock. It is what caught the one stage that had to be rolled back:
graphing the prompt-shaped `feat_encoder` call looked like a free win and
changed the output.

Identity is expected here by construction — a CUDA graph replays the same
kernels on the same weights and only changes the dispatch — so a difference is
a bug, not a tradeoff.

AND THE WARMUP HAS TO LAND BEFORE THE SEED

Captures happen on first use. One lands inside a measured render and that
render no longer compares to anything, because the allocator state and the
work queued ahead of it are not the same. Hence the throwaway render right
after the graft.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

CASES = [
    ("short, fr", "Ouais, je suis là. Carrément là."),
    ("long, fr", "Je ne sais pas, peut-être que c'est plus simple que ça. "
                 "Enfin, je vais regarder ce soir, quand tout sera calme."),
    ("english", "I don't know, maybe it's simpler than that."),
    ("spanish", "No sé, quizás es más simple que eso."),
    ("style instruction",
     "(ton sec et tendu, chaque mot appuyé)Il y a trois boîtes dans le "
     "couloir, et la fenêtre est restée ouverte."),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", required=True, help="VoxCPM2 checkpoint dir")
    ap.add_argument("--reference", required=True, help="a wav to clone from")
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    sys.path.insert(0, os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))

    import torch
    from voxcpm import VoxCPM

    import faster_voxcpm

    print(f"{torch.cuda.get_device_name(0)} - torch {torch.__version__} - "
          f"faster-voxcpm {faster_voxcpm.__version__}")
    model = VoxCPM.from_pretrained(args.weights)
    rate = int(getattr(model.tts_model, "sample_rate", 48000))

    def render(text):
        torch.manual_seed(args.seed)
        start = time.perf_counter()
        wav = model.generate(text=text, reference_wav_path=args.reference)
        torch.cuda.synchronize()
        return (np.asarray(wav, dtype=np.float32).reshape(-1),
                time.perf_counter() - start)

    torch.cuda.reset_peak_memory_stats()
    print("\nupstream", flush=True)
    before = {}
    for name, text in CASES:
        wav, took = render(text)
        before[name] = (wav, took)
        seconds = wav.size / rate
        print(f"  {name:18} {seconds:5.2f} s in {took:6.2f} s  "
              f"RTF {took / seconds:.3f}  ({seconds / took:.2f}x real time)",
              flush=True)
    peak_before = torch.cuda.max_memory_allocated() / 2 ** 20

    graft = faster_voxcpm.accelerate(model)
    render(CASES[0][1])  # throwaway: keep the captures out of a measured run
    if graft.failures:
        print(f"\n  CAPTURE FAILURES: {graft.failures}")
    torch.cuda.reset_peak_memory_stats()

    print("\nfaster-voxcpm", flush=True)
    gains, exact = [], 0
    for name, text in CASES:
        wav, took = render(text)
        ref, ref_took = before[name]
        same = ref.size == wav.size and bool(np.array_equal(ref, wav))
        exact += int(same)
        gains.append(ref_took / took)
        seconds = wav.size / rate
        note = "bit-identical" if same else (
            f"DIFFERENT ({ref.size} vs {wav.size} samples)"
            if ref.size != wav.size
            else f"DIFFERENT (max delta {np.abs(ref - wav).max():.3e})")
        print(f"  {name:18} {seconds:5.2f} s in {took:6.2f} s  "
              f"RTF {took / seconds:.3f}  ({seconds / took:.2f}x)  "
              f"{ref_took / took:.2f}x  {note}", flush=True)
    peak_after = torch.cuda.max_memory_allocated() / 2 ** 20

    print(f"\n  {exact}/{len(CASES)} bit-identical")
    print(f"  median {sorted(gains)[len(gains) // 2]:.2f}x "
          f"(min {min(gains):.2f}, max {max(gains):.2f})")
    print(f"  {graft.captures} capture(s), {graft.replays} replays, "
          f"{graft.fallbacks} fallbacks")
    print(f"  per entry point: {graft.breakdown()}")
    print(f"  peak memory {peak_before:,.0f} -> {peak_after:,.0f} MB "
          f"({peak_after - peak_before:+,.0f})")

    graft.undo()
    wav, took = render(CASES[0][1])
    ref, ref_took = before["short, fr"]
    print(f"\n  after undo(): {took:.2f} s (upstream {ref_took:.2f} s), "
          f"waveform restored "
          f"{ref.size == wav.size and bool(np.array_equal(ref, wav))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
