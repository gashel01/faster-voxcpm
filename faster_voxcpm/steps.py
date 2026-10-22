"""The three transformer stacks of a decode step, each in a CUDA graph.

WHAT THE PROFILE SAID, ONCE THE SAMPLER WAS ALREADY GRAPHED

    base_lm.forward_step   (28 layers, 2048 wide)   2236.1 ms   32 calls   44%
    flow head (graphed)                             1121.9 ms   33 calls   22%
    feat_encoder           (12 layers, 1024 wide)    794.7 ms   34 calls   16%
    residual_lm.forward_step (8 layers)              619.6 ms   32 calls   12%
    projections, stop head, the rest                  ~43 ms              ~0%

Seventy milliseconds to push ONE token through 28 layers. The weights of that
stack are 2807 MB, so reading them once costs about 9 ms at this card's
bandwidth — the other 60 are dispatch and Python. The 12-layer encoder is 396 MB
and takes 23 ms against a 1.3 ms floor; the 8-layer residual model is 720 MB and
takes 19 ms against 2.4 ms. All three are the same disease, and it is the one a
CUDA graph cures.

WHY THESE ARE CAPTURABLE AT ALL

Upstream already decodes with a `StaticKVCache`: a preallocated buffer, written
at `key_cache[:, :, position_id, :] = key_states` with `position_id` as a
tensor, and an attention mask built as `arange(...) <= position_id`. Every shape
is constant across the loop and only values change — which is exactly the shape
a graph wants. Nothing here had to be rewritten to make that true.

THE ONE HAZARD, AND HOW IT IS HANDLED

Capture is not free of side effects: the warmup passes and the capture itself
write keys into the cache at whatever position the static buffer holds. By the
time the first decode step runs, the prefill has already filled positions
0..n-1, and letting a capture scribble over position 0 would corrupt the
utterance silently.

So the cache is snapshotted before the capture and restored after it. It costs
one clone of a few tens of megabytes, once per process, and it removes an
entire class of "the first sentence after a restart is wrong" bug.
"""

from __future__ import annotations

import torch

from .capture import Captured, capture


class GraphedStep:
    """A graph for `forward_step(inputs_embeds, position_id)`.

    Falls back to the original call for any shape it was not built for — a
    voice engine that crashes is worth less than a slow one.
    """

    def __init__(self, model, original, label=""):
        self.model = model
        self.original = original
        self.label = label
        self._graphs: dict = {}
        self.captures = 0
        self.replays = 0
        self.fallbacks = 0
        self.failures: list[str] = []

    def _snapshot(self):
        """The KV cache as it stands, so a capture cannot damage it.

        Only the buffer is saved. `current_length` is a Python counter the
        *caller* advances through `kv_cache.step()`; a capture never touches
        it, so there is nothing to restore there.
        """
        cache = getattr(self.model, "kv_cache", None)
        buffer = getattr(cache, "kv_cache", None)
        if buffer is None:
            return None
        return buffer, buffer.clone()

    @staticmethod
    def _restore(snapshot) -> None:
        if snapshot is not None:
            buffer, saved = snapshot
            buffer.copy_(saved)

    def _build(self, embeds, position_id) -> Captured:
        embeds_s = torch.zeros_like(embeds)
        position_s = torch.zeros_like(position_id)
        snapshot = self._snapshot()
        try:
            graph, output, context = capture(
                lambda: self.original(embeds_s, position_s))
        finally:
            # Restored whether the capture worked or not: a half-finished
            # capture has still written into the cache.
            self._restore(snapshot)
        self.captures += 1
        return Captured(graph=graph, output=output,
                        inputs={"embeds": embeds_s, "position": position_s},
                        held={"context": context})

    def __call__(self, inputs_embeds, position_id):
        key = (tuple(inputs_embeds.shape), str(inputs_embeds.dtype),
               tuple(position_id.shape))
        entry = self._graphs.get(key)
        if entry is None:
            try:
                entry = self._build(inputs_embeds, position_id)
            except Exception as exc:  # pragma: no cover - driver dependent
                self.failures.append(f"{self.label}: {exc!r}")
                entry = False
            self._graphs[key] = entry
        if entry is False:
            self.fallbacks += 1
            return self.original(inputs_embeds, position_id)
        self.replays += 1
        return entry.replay(embeds=inputs_embeds, position=position_id)


class GraphedCall:
    """A graph for a module with one tensor in and one tensor out.

    Used for `feat_encoder`, which the loop calls once per step on the latest
    patch — shape (1, 1, 4, 64) every time. It takes the *bound call*, not the
    module: `nn.Module` refuses to have anything but a module assigned over a
    registered child, so the graft replaces `forward`, and holding the original
    call is also what keeps the fallback from recursing into itself.

    GRAPH ONLY WHAT REPEATS, and `hot` is how that is enforced.

    `feat_encoder` is called with two shapes: once per utterance on the whole
    audio prompt — 60 patches — and once per decode step on a single patch.
    Fifteen of sixteen calls are the single-patch one, so the prompt call is
    worth nothing to graph. It is also worth *less* than nothing: capturing it
    was measured to change the final waveform, while capturing only the hot
    shape is identical to the bit.

        all shapes graphed      DIFFERENT (99840 vs 115200 samples)
        hot shape only          identical to the bit
        prompt shape only       DIFFERENT (99840 vs 115200 samples)

    The cause was not found, and three hypotheses were tested and refuted
    rather than assumed: the graph is numerically exact on arbitrary inputs at
    both shapes, a capture does not shift the global RNG stream, and neither
    does a replay. So this is a rule backed by measurement, not an explanation.
    It happens to be the right rule anyway — a shape seen once per utterance
    does not deserve a graph, and is exactly where side effects hide.
    """

    def __init__(self, call, label="", hot=None):
        self.call = call
        self.label = label
        self.hot = hot
        self._graphs: dict = {}
        self.captures = 0
        self.replays = 0
        self.fallbacks = 0
        self.failures: list[str] = []

    def _build(self, x) -> Captured:
        x_s = torch.zeros_like(x)
        graph, output, context = capture(lambda: self.call(x_s))
        self.captures += 1
        return Captured(graph=graph, output=output, inputs={"x": x_s},
                        held={"context": context})

    def __call__(self, x):
        if self.hot is not None and not self.hot(x):
            self.fallbacks += 1
            return self.call(x)
        key = (tuple(x.shape), str(x.dtype))
        entry = self._graphs.get(key)
        if entry is None:
            try:
                entry = self._build(x)
            except Exception as exc:  # pragma: no cover - driver dependent
                self.failures.append(f"{self.label}: {exc!r}")
                entry = False
            self._graphs[key] = entry
        if entry is False:
            self.fallbacks += 1
            return self.call(x)
        self.replays += 1
        return entry.replay(x=x)


__all__ = ["GraphedStep", "GraphedCall"]
