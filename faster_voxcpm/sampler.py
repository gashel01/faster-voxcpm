"""The flow-matching sampler in one CUDA graph. Two thirds of the time is here.

WHAT THE PROFILE SAID, BEFORE ANY OF THIS WAS WRITTEN

On an RTX 3070 Ti Laptop, 2.88 s of audio in 5.62 s (RTF 1.95):

    flow head, 10 steps     3632.7 ms   18 calls   65%   201.8 ms/call
    everything else         1991.1 ms                35%

201.8 ms for 10 flow steps over 12 DiT layers at batch 2 is 0.84 ms per layer
pass, on tensors of shape (2, 64, 4). The arithmetic there is nothing. What
gets paid is the *dispatch* of a few thousand kernels, on a machine where a
trivial kernel launch costs 12.4 microseconds against Linux's 5-10, because
Windows runs the WDDM driver model. Quantization would not touch any of it.

The same diagnosis on Ming-3.0-tts took that component from 167 ms to 20 ms
with a CUDA graph, output identical to the bit — which is also the proof: a
graph changes no kernel, no weight and no shape, only the dispatch.

WHY ONE GRAPH COVERS A WHOLE UTTERANCE

Every shape into `UnifiedCFM.forward` is constant across the autoregressive
loop. `mu` is one hidden vector per step, and `cond` is `prefix_feat_cond`,
which the loop reassigns to the *latest patch only* — `prefix_feat_cond =
pred_feat` — so it never grows. `patch_size` is 4 and `n_timesteps` is 10.

WHAT STAYS OUTSIDE THE CAPTURE, AND WHY

  - the noise draw. It stays in the global RNG stream so a fixed seed still
    yields the same voice; the order, shape and dtype of that draw are what
    makes a seed reproducible.
  - `t_span`, built by `torch.linspace` and then bent by the sway coefficient.
    It depends only on the step count, so it is a constant — built once,
    outside, and *held*, because a constant left in a local is exactly the bug
    described in `capture.py`.

Everything else is the upstream `solve_euler` loop, called as it stands, so
nothing in the model is rewritten and nothing can drift numerically.
"""

from __future__ import annotations

import torch

from .capture import Captured, capture


class GraphedSampler:
    """Drop-in replacement for `UnifiedCFM.forward` with identical output.

    Installed over the module's `forward`, and it falls back to the original
    call for any shape or setting it was not built for — a voice engine that
    crashes is worth less than a slow one.
    """

    def __init__(self, cfm, original):
        self.cfm = cfm
        self.original = original
        self._graphs: dict = {}
        self.captures = 0
        self.replays = 0
        self.fallbacks = 0
        self.failures: list[str] = []

    def _t_span(self, n_timesteps, sway, device, dtype):
        """`t_span` exactly as upstream builds it, built outside the capture."""
        t_span = torch.linspace(1, 0, n_timesteps + 1, device=device,
                                dtype=dtype)
        return t_span + sway * (torch.cos(torch.pi / 2 * t_span) - 1 + t_span)

    def _build(self, key, mu, cond, patch_size, cfg_value, use_cfg_zero_star,
               n_timesteps, sway) -> Captured:
        b = mu.shape[0]
        x = torch.zeros((b, self.cfm.in_channels, patch_size),
                        device=mu.device, dtype=mu.dtype)
        mu_s = torch.zeros_like(mu)
        cond_s = torch.zeros_like(cond)
        t_span = self._t_span(n_timesteps, sway, mu.device, mu.dtype)

        def body():
            return self.cfm.solve_euler(
                x=x, t_span=t_span, mu=mu_s, cond=cond_s,
                cfg_value=cfg_value, use_cfg_zero_star=use_cfg_zero_star)

        graph, output, context = capture(body)
        self.captures += 1
        # `t_span` is held on purpose. It is a constant, the graph reads it,
        # and a constant left in a local is the bug in `capture.py`.
        return Captured(graph=graph, output=output,
                        inputs={"x": x, "mu": mu_s, "cond": cond_s},
                        held={"t_span": t_span, "context": context})

    def __call__(self, mu, n_timesteps, patch_size, cond, temperature=1.0,
                 cfg_value=1.0, sway_sampling_coef=1.0,
                 use_cfg_zero_star=True):
        # The draw stays here, in this order, shape and dtype — that is what
        # keeps a fixed seed equivalent to the original implementation's.
        b = mu.shape[0]
        z = torch.randn((b, self.cfm.in_channels, patch_size),
                        device=mu.device, dtype=mu.dtype) * temperature

        key = (tuple(mu.shape), tuple(cond.shape), str(mu.dtype),
               int(n_timesteps), int(patch_size), float(cfg_value),
               float(sway_sampling_coef), bool(use_cfg_zero_star))
        entry = self._graphs.get(key)
        if entry is None:
            try:
                entry = self._build(key, mu, cond, patch_size, cfg_value,
                                    use_cfg_zero_star, n_timesteps,
                                    sway_sampling_coef)
            except Exception as exc:  # pragma: no cover - driver dependent
                self.failures.append(repr(exc))
                entry = False
            self._graphs[key] = entry
        if entry is False:
            self.fallbacks += 1
            t_span = self._t_span(n_timesteps, sway_sampling_coef, mu.device,
                                  mu.dtype)
            return self.cfm.solve_euler(
                x=z, t_span=t_span, mu=mu, cond=cond, cfg_value=cfg_value,
                use_cfg_zero_star=use_cfg_zero_star)

        self.replays += 1
        return entry.replay(x=z, mu=mu, cond=cond)


__all__ = ["GraphedSampler"]
