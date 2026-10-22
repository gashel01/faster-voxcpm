"""CUDA graph capture, and the one rule that makes it safe.

A CUDA graph records *addresses*, not values. Every tensor a captured graph
reads must therefore stay alive for as long as the graph does, and that
includes the constants. Miss one and you get the most misleading bug in this
area:

    the graph is exact right after capture, and wrong later.

This is not a hypothetical. The sibling project for Ming-3.0-tts kept the flow
sampler's timestep tensor in a local variable of the build function. When that
function returned, the tensor was collected, its device memory went back to the
caching allocator, and the next large allocation wrote over it — so the graph
replayed with arbitrary timesteps. In isolation every test passed; inside the
real decode loop the output diverged at the very first step, by a magnitude
that said "wrong input", not "rounding". Two benches chased the wrong suspects
first, the cuBLAS workspace and a rotary-frequency cache, and both cleared CUDA
of any wrongdoing.

So `Captured` holds the graph *and* everything it touches, and nothing a graph
reads is ever left in a local.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict

import torch


@dataclass
class Captured:
    """A replayable graph together with every tensor it reads or writes.

    `inputs` are the static buffers a caller writes into before each replay,
    `output` is the static buffer the graph writes, and `held` is for anything
    else the graph reads — constants, precomputed tables, the capture context.
    Nothing here is decoration: dropping any of it reintroduces the bug in this
    module's docstring.
    """

    graph: torch.cuda.CUDAGraph
    output: Any
    inputs: Dict[str, torch.Tensor] = field(default_factory=dict)
    held: Dict[str, Any] = field(default_factory=dict, repr=False)

    def replay(self, **values: torch.Tensor):
        """Copy `values` into the static inputs, replay, return a fresh output.

        The output is cloned because the next replay overwrites that buffer and
        callers keep their results.
        """
        for name, value in values.items():
            self.inputs[name].copy_(value)
        self.graph.replay()
        return self.output.clone()


def capture(body: Callable[[], Any], warmups: int = 3):
    """Warm up on a side stream, then capture `body`. Returns (graph, output, ctx).

    The side-stream warmup is required by CUDA, not optional hygiene: it is what
    forces cuBLAS to allocate its workspaces and pick its algorithms, neither of
    which can happen during a capture.

    The capture context is returned so the caller can hold it; see this module's
    docstring for why nothing is left to go out of scope.
    """
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(warmups):
            body()
    torch.cuda.current_stream().wait_stream(stream)

    graph = torch.cuda.CUDAGraph()
    context = torch.cuda.graph(graph)
    with context:
        output = body()
    return graph, output, context


__all__ = ["Captured", "capture"]
