"""What does 4-bit buy, and what does it cost?

THE ACCEPTANCE TEST CHANGES HERE, AND THAT IS THE POINT

Every other bench in this folder checks the waveform to the bit, because
nothing else in this package changes the computation. Quantization does. So
this one reports three things and judges none of them:

    memory   the objective — measured as the process sees it, not as torch
             reports it, because `max_memory_allocated` ignores the CUDA
             context, the cuBLAS workspaces and the allocator reserve
    speed    a bonus, and not assumed: int8 on a sibling model once bought
             38 MB for half the throughput
    audio    written out, because whether 4 bits went too far is a question
             for an ear

ONE MODE PER PROCESS, ON PURPOSE

Running both configurations in one process makes the memory number
meaningless: the first model's allocator reserve never goes back to the
driver, so the second looks bigger than it is. So this takes `--mode` and is
run twice. It costs a model load; it buys a number you can believe.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
import wave

import numpy as np

CASES = [
    ("short-fr", "Ouais, je suis là. Carrément là."),
    ("long-fr", "Je ne sais pas, peut-être que c'est plus simple que ça. "
                "Enfin, je vais regarder ce soir, quand tout sera calme."),
    ("english", "I don't know, maybe it's simpler than that."),
    ("spanish", "No sé, quizás es más simple que eso."),
    ("instruction", "(ton sec et tendu, chaque mot appuyé)Il y a trois boîtes "
                    "dans le couloir, et la fenêtre est restée ouverte."),
]

COUNTER = ("$c = Get-Counter '\\GPU Process Memory(*)\\Dedicated Usage'; "
           "($c.CounterSamples | Where-Object {{ $_.InstanceName -like "
           "'*{pid}*' }} | Measure-Object -Sum CookedValue).Sum / 1MB")


def process_vram() -> float:
    """What this process holds on the card, as the OS sees it.

    Not `torch.cuda.max_memory_allocated`: that counts what torch's allocator
    asked for and ignores the CUDA context, the cuBLAS workspaces and the
    reserve. On a sibling model the gap was 566 MB, and the whole point of this
    bench is the number that decides whether the model fits beside something
    else.
    """
    if sys.platform != "win32":
        return float("nan")
    try:
        out = subprocess.run(  # noqa: S603
            ["powershell", "-NoProfile", "-Command",
             COUNTER.format(pid=os.getpid())],
            capture_output=True, text=True, timeout=60, check=True).stdout
        return float(out.strip() or 0)
    except Exception:  # noqa: BLE001 - a missing counter is not a fault
        return float("nan")


def write_wav(path, x, rate):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    x = np.clip(np.asarray(x, dtype=np.float32).reshape(-1), -1.0, 1.0)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes((x * 32767).astype("<i2").tobytes())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--reference", required=True)
    ap.add_argument("--out", required=True, help="folder for the wav files")
    ap.add_argument("--mode", default="bf16", choices=("bf16", "nf4", "fp4"),
                    help="one configuration per process; run it twice")
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    sys.path.insert(0, os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))

    import torch
    from voxcpm import VoxCPM

    import faster_voxcpm
    from faster_voxcpm.quantize import quantize_language_models

    print(f"{torch.cuda.get_device_name(0)} - torch {torch.__version__}")

    model = VoxCPM.from_pretrained(args.weights)
    rate = int(getattr(model.tts_model, "sample_rate", 48000))

    print(f"\nmode {args.mode}", flush=True)
    if args.mode != "bf16":
        # Before the graphs: they capture whatever the model does at capture
        # time, and quantizing afterwards would leave them replaying kernels
        # for weights that no longer exist.
        report = quantize_language_models(model, quant_type=args.mode)
        for name, r in report.items():
            print(f"  {name:14} {r['layers']:3d} layers  "
                  f"{r['before_mb']:7,.0f} -> {r['after_mb']:7,.0f} MB")

    graft = faster_voxcpm.accelerate(model)

    def render(text):
        torch.manual_seed(args.seed)
        start = time.perf_counter()
        wav = model.generate(text=text, reference_wav_path=args.reference)
        torch.cuda.synchronize()
        return (np.asarray(wav, dtype=np.float32).reshape(-1),
                time.perf_counter() - start)

    render(CASES[0][1])  # throwaway: captures land before the measurements
    if graft.failures:
        print(f"  CAPTURE FAILURES: {graft.failures[0][:200]}")

    rates = []
    for name, text in CASES:
        wav, took = render(text)
        seconds = wav.size / rate
        rates.append(seconds / took)
        write_wav(os.path.join(args.out, args.mode, f"{name}.wav"), wav, rate)
        print(f"  {name:14} {seconds:5.2f} s in {took:6.2f} s  "
              f"RTF {took / seconds:.3f}  ({seconds / took:.2f}x)", flush=True)

    print(f"\n  process VRAM {process_vram():,.0f} MB   "
          f"torch peak {torch.cuda.max_memory_allocated() / 2**20:,.0f} MB")
    print(f"  {graft.captures} captures, {graft.replays} replays, "
          f"{graft.fallbacks} fallbacks")
    print(f"  median {sorted(rates)[len(rates) // 2]:.2f}x real time")
    print(f"  wav files in {os.path.join(args.out, args.mode)}")
    if args.mode != "bf16":
        print("  the waveform is NOT expected to match bf16 — listen before "
              "deciding")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
