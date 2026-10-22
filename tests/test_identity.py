"""The waveform must not move. That is the whole contract.

These tests need a GPU, the checkpoint and a reference wav, so they are skipped
unless two environment variables point at them:

    FASTER_VOXCPM_WEIGHTS     a VoxCPM2 checkpoint directory
    FASTER_VOXCPM_REFERENCE   a wav file to clone from

Skipping rather than failing is deliberate: the suite should be runnable by
someone reading the code who has no 4.7 GB checkpoint on disk, and a test that
cannot run is not a test that failed.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

WEIGHTS = os.environ.get("FASTER_VOXCPM_WEIGHTS")
REFERENCE = os.environ.get("FASTER_VOXCPM_REFERENCE")
SEED = 123

needs_model = pytest.mark.skipif(
    not (WEIGHTS and REFERENCE),
    reason="set FASTER_VOXCPM_WEIGHTS and FASTER_VOXCPM_REFERENCE to run",
)


@pytest.fixture(scope="module")
def model():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    from voxcpm import VoxCPM

    return VoxCPM.from_pretrained(WEIGHTS)


def render(model, text):
    import torch

    torch.manual_seed(SEED)
    wav = model.generate(text=text, reference_wav_path=REFERENCE)
    torch.cuda.synchronize()
    return np.asarray(wav, dtype=np.float32).reshape(-1)


@needs_model
@pytest.mark.parametrize("text", [
    pytest.param("Ouais, je suis là. Carrément là.", id="short-fr"),
    pytest.param("I don't know, maybe it's simpler than that.", id="english"),
    pytest.param("(ton sec et tendu)Il y a trois boîtes dans le couloir.",
                 id="style-instruction"),
])
def test_waveform_is_bit_identical(model, text):
    """Grafted output must equal upstream output, sample for sample.

    Not "close enough": a graph replays the same kernels on the same weights,
    so any difference at all is a bug rather than a tradeoff. This is the test
    that caught the prompt-shaped `feat_encoder` capture.
    """
    import faster_voxcpm

    upstream = render(model, text)
    graft = faster_voxcpm.accelerate(model)
    try:
        assert not graft.failures, graft.failures
        render(model, text)  # throwaway, so captures land before the measure
        grafted = render(model, text)
    finally:
        graft.undo()

    assert grafted.size == upstream.size
    assert np.array_equal(upstream, grafted), (
        f"max delta {np.abs(upstream - grafted).max():.3e}")


@needs_model
def test_undo_restores_upstream(model):
    """After undo() the model must behave exactly as if nothing happened."""
    import faster_voxcpm

    text = "Ouais, je suis là. Carrément là."
    upstream = render(model, text)
    faster_voxcpm.accelerate(model).undo()
    assert np.array_equal(upstream, render(model, text))


@needs_model
def test_the_prompt_shaped_encoder_call_is_not_graphed(model):
    """Graph only what repeats — and here, only what was measured to be safe.

    `feat_encoder` sees two shapes: the whole audio prompt once per utterance,
    and one patch per decode step. Capturing the first changed the waveform,
    for a reason that was looked for and not found. So it falls back, and that
    has to stay true.
    """
    import faster_voxcpm

    graft = faster_voxcpm.accelerate(model)
    try:
        render(model, "Ouais, je suis là.")
        encoder = next(g for _, _, g in graft.parts
                       if type(g).__name__ == "GraphedCall")
        shapes = {key[0] for key in encoder._graphs}
        assert shapes, "the hot shape was never graphed"
        assert all(shape[1] == 1 for shape in shapes), shapes
        assert encoder.fallbacks >= 1, "the prompt call should have fallen back"
    finally:
        graft.undo()


def test_captured_holds_everything():
    """A graph reads addresses, so `Captured` must keep its tensors alive.

    Runs without a GPU: it checks the dataclass contract, which is the one
    thing that made the sibling project for Ming silently wrong for an hour.
    """
    from faster_voxcpm.capture import Captured

    fields = Captured.__dataclass_fields__
    assert "held" in fields, "constants a graph reads must be held somewhere"
    assert "inputs" in fields and "output" in fields
