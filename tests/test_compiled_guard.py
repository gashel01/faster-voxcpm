"""accelerate() refuses a model upstream's optimize() already compiled.

No checkpoint needed: the guard only looks at which wrappers are in place,
so stand-ins with the same attributes are enough. These are the markers
`torch.compile` leaves: `_torchdynamo_orig_callable` on a compiled function,
`_orig_mod` on a compiled module, and upstream's own `_feat_encoder_raw`.
"""

from __future__ import annotations

import types

import pytest

from faster_voxcpm import _compiled_entry_points, accelerate


def stand_in(compiled: bool):
    def forward_step(*args, **kwargs):
        return None

    if compiled:
        forward_step._torchdynamo_orig_callable = forward_step
    tts = types.SimpleNamespace(
        base_lm=types.SimpleNamespace(forward_step=forward_step),
        residual_lm=types.SimpleNamespace(forward_step=forward_step),
        feat_decoder=types.SimpleNamespace(
            estimator=types.SimpleNamespace(**({"_orig_mod": object()}
                                               if compiled else {}))),
    )
    if compiled:
        tts._feat_encoder_raw = object()
    return types.SimpleNamespace(tts_model=tts)


def test_eager_model_has_no_compiled_entry_points():
    assert _compiled_entry_points(stand_in(False).tts_model) == []


def test_compiled_model_is_detected_everywhere():
    assert _compiled_entry_points(stand_in(True).tts_model) == [
        "base_lm.forward_step", "residual_lm.forward_step", "feat_encoder",
        "feat_decoder.estimator"]


def test_accelerate_refuses_compiled_model_and_names_the_fix():
    with pytest.raises(RuntimeError, match="optimize=False"):
        accelerate(stand_in(True))
