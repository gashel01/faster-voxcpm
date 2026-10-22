"""Weight quantization for the two language models. Memory first.

THIS IS THE ONE THING HERE THAT CHANGES THE OUTPUT

Everything else in this package is bit-identical by construction: a CUDA graph
replays the same kernels on the same weights. Quantization does not. It stores
different numbers, so the waveform moves, and no bench can tell you whether it
moved too far. Only listening can. Treat it accordingly — it is a separate
call, not part of `accelerate`.

WHY THE TWO LANGUAGE MODELS AND NOTHING ELSE

Read off the checkpoint, not guessed:

    base_lm        2807 MB   64.3%
    residual_lm     720 MB   16.5%
    feat_decoder    404 MB    9.3%   <- the flow head
    feat_encoder    396 MB    9.1%
    audio VAE       359 MB
    the rest        ~40 MB

81% of the weights sit in the two language models. The flow head — the part
that carries the voice, and the part the graphs accelerate — is 9%. So the
separation is ideal: compress the big dull part, leave the small delicate part
in bf16.

That is also what the most thoughtful community GGUF build does
(`voxcpm2-q8_0-locdit-f16.gguf`), and its saving, 1856 MB against the f16
build, is exactly what 8-bit on these two stacks predicts.

NF4 RATHER THAN INT8, BECAUSE THE GOAL IS MEMORY

Four-bit NF4 divides the stored weights by four: the two stacks go from
3527 MB to about 880 MB. Int8 would halve them instead. With memory as the
objective and speed as a bonus, four bits is the stronger play — and there is
a measured reason to distrust int8 as a *speed* move anyway: on Ming-3.0-tts
it bought 38 MB for half the throughput.

ORDER MATTERS: QUANTIZE, THEN GRAPH

`accelerate` captures whatever the model does at capture time. Quantizing
afterwards would leave the graphs replaying kernels for weights that no longer
exist. Quantize first.
"""

from __future__ import annotations

import torch

# What is quantized, and what is deliberately left alone.
LANGUAGE_MODELS = ("base_lm", "residual_lm")


def _quantized_linear(linear, quant_type, compute_dtype):
    """One `nn.Linear` as a 4-bit one, carrying its weights across."""
    import bitsandbytes as bnb
    from bitsandbytes.nn import Linear4bit, Params4bit

    replacement = Linear4bit(
        linear.in_features, linear.out_features,
        bias=linear.bias is not None,
        compute_dtype=compute_dtype, quant_type=quant_type,
    )
    # Through the CPU on purpose: `Params4bit` quantizes on its first move to
    # the device, so it has to start somewhere else.
    replacement.weight = Params4bit(
        linear.weight.data.detach().to("cpu", torch.float16),
        requires_grad=False, quant_type=quant_type,
    )
    if linear.bias is not None:
        replacement.bias = torch.nn.Parameter(
            linear.bias.data.detach().clone(), requires_grad=False)
    del bnb
    return replacement.to(linear.weight.device)


def quantize_language_models(model, quant_type: str = "nf4",
                             compute_dtype=torch.bfloat16,
                             stacks=LANGUAGE_MODELS) -> dict:
    """Replace every `nn.Linear` inside the language models with a 4-bit one.

    Returns what it did, per stack: the number of layers and the megabytes
    before and after, so a bench can report the saving rather than assume it.

    Nothing outside `stacks` is touched — not the flow head, not the encoder,
    not the VAE, not the projections.
    """
    tts = getattr(model, "tts_model", model)
    report: dict[str, dict] = {}

    for name in stacks:
        stack = getattr(tts, name, None)
        if stack is None:
            continue
        before = sum(p.numel() * p.element_size() for p in stack.parameters())
        swapped = 0
        # Collected first: replacing while walking mutates what we walk.
        targets = [(parent_name, child_name, child)
                   for parent_name, parent in stack.named_modules()
                   for child_name, child in parent.named_children()
                   if isinstance(child, torch.nn.Linear)]
        for parent_name, child_name, child in targets:
            parent = stack.get_submodule(parent_name) if parent_name else stack
            setattr(parent, child_name,
                    _quantized_linear(child, quant_type, compute_dtype))
            swapped += 1
        after = sum(p.numel() * p.element_size() for p in stack.parameters())
        report[name] = {
            "layers": swapped,
            "before_mb": before / 2 ** 20,
            "after_mb": after / 2 ** 20,
        }

    torch.cuda.empty_cache()
    return report


__all__ = ["quantize_language_models", "LANGUAGE_MODELS"]
