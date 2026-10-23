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

FOUR BITS OR EIGHT, AND THE ANSWER CHANGED ONCE IT WAS HEARD

Four-bit NF4 divides the stored weights by four: the two stacks go from
3527 MB to about 880 MB. Int8 halves them instead. With memory as the
objective, four bits was the obvious play, and on neutral sentences the two
were judged indistinguishable.

**With a delivery instruction they are not.** The same neutral sentence under
six instructions moved its duration by 1.92 s in bf16 and only 1.12 s in NF4 —
and the ear agreed that bf16 "joue plus". Four bits appears to flatten the
acting, which is the one thing this engine is being kept for.

So both are offered, `quant_type="int8"` gives back about half of what four
bits saved, and the choice is a listening decision rather than a number.

ORDER MATTERS: QUANTIZE, THEN GRAPH

`accelerate` captures whatever the model does at capture time. Quantizing
afterwards would leave the graphs replaying kernels for weights that no longer
exist. Quantize first.
"""

from __future__ import annotations

import torch

# What is quantized, and what is deliberately left alone.
LANGUAGE_MODELS = ("base_lm", "residual_lm")


def _int8_linear(linear):
    """One `nn.Linear` as an 8-bit one.

    Here because four bits was heard to flatten the delivery — the same
    sentence under six instructions moved its duration by 1.92 s in bf16 and
    only 1.12 s in NF4, and the ear agreed. Eight bits gives back about half
    the memory four bits saved and keeps more of the amplitude, which is the
    trade this engine is being kept for in the first place.
    """
    from bitsandbytes.nn import Int8Params, Linear8bitLt

    replacement = Linear8bitLt(
        linear.in_features, linear.out_features,
        bias=linear.bias is not None, has_fp16_weights=False,
    )
    # Through the CPU on purpose: `Int8Params` quantizes on its first move to
    # the device, so it has to start somewhere else.
    replacement.weight = Int8Params(
        linear.weight.data.detach().to("cpu", torch.float16),
        requires_grad=False, has_fp16_weights=False,
    )
    if linear.bias is not None:
        replacement.bias = torch.nn.Parameter(
            linear.bias.data.detach().clone(), requires_grad=False)
    return replacement.to(linear.weight.device)


def _quantized_linear(linear, quant_type, compute_dtype):
    """One `nn.Linear`, compressed. Four bits by default, eight on request."""
    if quant_type == "int8":
        return _int8_linear(linear)

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
