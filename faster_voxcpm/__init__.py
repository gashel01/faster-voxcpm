"""faster-voxcpm: VoxCPM2, same weights, same output, with the dispatch removed.

    from voxcpm import VoxCPM
    from faster_voxcpm import accelerate

    model = VoxCPM.from_pretrained("/path/to/VoxCPM2")
    accelerate(model)              # that is the whole integration

    wav = model.generate(text="...", reference_wav_path="reference.wav")

Nothing here retrains, quantizes or rewrites the model. The flow-matching
sampler — 65% of a turn, measured — is captured in a CUDA graph and replayed.
A graph changes no kernel, no weight and no shape, only the dispatch, so the
waveform must come back identical to the bit, and the bench checks that before
it looks at the clock.

Why it is worth doing: a decode step runs 10 flow steps over 12 DiT layers at
batch 2, on tensors of shape (2, 64, 4). The arithmetic is nothing; what gets
paid is a few thousand kernel launches, at 12.4 microseconds each on this
machine against Linux's 5-10, because Windows runs the WDDM driver model.

Requires: `voxcpm` installed, torch with CUDA.
"""

from __future__ import annotations

from .capture import Captured, capture
from .sampler import GraphedSampler
from .steps import GraphedCall, GraphedStep

__version__ = "0.1.0"


class Graft:
    """What was replaced, and how to put it back.

    The graft only ever touches an entry point, never upstream's code, so it
    undoes cleanly — which is what lets a bench compare both paths inside one
    process on one card in one state. That is the only way to prove the output
    did not move.
    """

    def __init__(self, parts):
        # (owner, attribute, grafted object) for each entry point replaced.
        self.parts = parts

    @property
    def sampler(self):
        for _, _, graphed in self.parts:
            if isinstance(graphed, GraphedSampler):
                return graphed
        return None

    @property
    def replays(self) -> int:
        return sum(g.replays for _, _, g in self.parts)

    @property
    def captures(self) -> int:
        return sum(g.captures for _, _, g in self.parts)

    @property
    def fallbacks(self) -> int:
        return sum(g.fallbacks for _, _, g in self.parts)

    @property
    def failures(self) -> list:
        return [f for _, _, g in self.parts for f in g.failures]

    def breakdown(self) -> dict:
        """Replays per entry point, for a bench that wants to see the split.

        Keyed by the graft's own label when it has one, because two of the
        entry points are `MiniCPMModel.forward_step` — `base_lm` and
        `residual_lm` — and keying by class name silently collapsed them into
        one row that under-reported the work by half.
        """
        return {(getattr(g, "label", "") or f"{type(o).__name__}.{a}"): g.replays
                for o, a, g in self.parts}

    def undo(self) -> None:
        for owner, attribute, _ in self.parts:
            # Dropping the instance attribute lets the class method, or the
            # bound method, take over again.
            try:
                delattr(owner, attribute)
            except AttributeError:  # pragma: no cover - already restored
                pass


def _find_cfm(model):
    """Locate the flow-matching module, whatever it is hanging off.

    By class rather than by attribute path: the attribute layout changes
    between releases, the class much less. Upstream calls it `feat_decoder`
    today.
    """
    from voxcpm.modules.locdit.unified_cfm import UnifiedCFM

    root = getattr(model, "tts_model", model)
    for module in root.modules():
        if isinstance(module, UnifiedCFM):
            return module
    raise RuntimeError("no UnifiedCFM in this model — nothing to accelerate")


STACKS = ("base_lm", "residual_lm", "feat_encoder")


def accelerate(model, sampler: bool = True, stacks=STACKS) -> Graft:
    """Graft the graphs onto an already-loaded VoxCPM2.

    `model` is the `VoxCPM` wrapper or the inner TTS model; either works.
    Four entry points are replaced, in the order the profile ranked them:

        base_lm.forward_step      44% of a turn after the sampler was done
        the flow sampler          65% before it was, 22% after
        feat_encoder              16%
        residual_lm.forward_step  12%

    Each stage can be turned off to isolate it in a bench — `stacks` takes a
    subset of `STACKS`, or `False` for none — which is how the one that broke
    bit identity was found. Returns a `Graft`; call `undo()` to restore every
    entry point.
    """
    wanted = set(stacks or ())
    tts = getattr(model, "tts_model", model)
    parts = []

    if sampler:
        cfm = _find_cfm(model)
        graphed = GraphedSampler(cfm, cfm.forward)
        cfm.forward = graphed
        parts.append((cfm, "forward", graphed))

    for name in ("base_lm", "residual_lm"):
        if name in wanted:
            stack = getattr(tts, name, None)
            if stack is None or not hasattr(stack, "forward_step"):
                continue
            graphed = GraphedStep(stack, stack.forward_step, label=name)
            stack.forward_step = graphed
            parts.append((stack, "forward_step", graphed))

    if "feat_encoder" in wanted:
        encoder = getattr(tts, "feat_encoder", None)
        if encoder is not None:
            # Only the single-patch call, which is the decode step. The
            # prompt call is one of sixteen and capturing it was measured to
            # change the waveform — see `GraphedCall`.
            graphed = GraphedCall(encoder.forward, label="feat_encoder",
                                  hot=lambda x: x.shape[1] == 1)
            encoder.forward = graphed
            parts.append((encoder, "forward", graphed))

    return Graft(parts)


__all__ = ["accelerate", "Graft", "STACKS", "GraphedSampler", "GraphedStep",
           "GraphedCall", "Captured", "capture", "__version__"]
