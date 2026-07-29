# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#2026.7.29 19:46
import contextlib
from collections.abc import Callable

import torch
import torch.fx.traceback as fx_traceback
import torch.nn as nn
from torch._subclasses.fake_tensor import FakeTensorMode

from torchtitan.config import CompileConfig
from torchtitan.tools.logging import logger


# TODO: Remove this monkeypatch once FakeTensorMode.__init__ is decorated with
# @torch.compiler.disable(recursive=True) upstream.
# See https://github.com/pytorch/pytorch/issues/178887
FakeTensorMode.__init__ = torch.compiler.disable(  # type: ignore[method-assign]
    FakeTensorMode.__init__, recursive=True
)


# Toggled on by ``_maybe_regional_inductor_backend`` when the model is compiled
# with a non-inductor backend that needs inductor-only regions (e.g.
# FlexAttention) scooped into an inductor sub-compile. Read by
# ``maybe_regional_inductor`` at trace time; left False on the default inductor
# / eager paths so no annotation metadata is emitted.
_regional_inductor_enabled: bool = False


def apply_compile(
    model: nn.Module,
    compile_config: CompileConfig,
    *,
    layer_filter: Callable[[str, nn.Module], bool] | None = None,
) -> None:
    """
    Apply torch.compile to selected TransformerBlocks.

    By default, every block under ``model.layers`` is compiled. ``layer_filter``
    can keep unsupported blocks in eager mode. The callback receives the layer
    ID and the underlying TransformerBlock. If activation checkpointing has
    already wrapped the block, this function unwraps
    ``_checkpoint_wrapped_module`` for filtering while compiling the outer
    checkpoint wrapper itself.
    """
    torch._dynamo.config.capture_scalar_outputs = True
    torch._dynamo.config.skip_fwd_side_effects_in_bwd_under_checkpoint = (
        True  # pyrefly: ignore [bad-assignment]
    )

    backend = _maybe_regional_inductor_backend(model, compile_config.backend)

    compiled_layers: list[str] = []
    skipped_layers: list[str] = []

    # pyrefly: ignore [missing-attribute]
    for layer_id, transformer_block in model.layers.named_children():
        filter_target = getattr(
            transformer_block,
            "_checkpoint_wrapped_module",
            transformer_block,
        )

        if layer_filter is not None and not layer_filter(layer_id, filter_target):
            skipped_layers.append(layer_id)
            continue

        transformer_block.compile(backend=backend, fullgraph=True)
        compiled_layers.append(layer_id)

    if not compiled_layers:
        logger.warning(
            "torch.compile did not select any TransformerBlock layers; "
            "skipped layers=%s",
            skipped_layers,
        )
        return

    logger.info(
        "Compiled TransformerBlock layers=%s with backend=%s; skipped layers=%s",
        compiled_layers,
        compile_config.backend,
        skipped_layers,
    )


def _maybe_regional_inductor_backend(model: nn.Module, backend: str) -> str | Callable:
    """Wrap the ``aot_eager`` backend so inductor-only flex regions are scooped out.

    ``regional_inductor`` lowers just the regions annotated with ``compile_with_inductor`` (see
    ``FlexAttention.forward``) to inductor while the rest stays in aot_eager.

    Only applied for ``aot_eager`` on models that actually use FlexAttention, so
    dense/non-flex aot_eager paths are left untouched. Other non-inductor backends
    can't be scooped here and raise rather than silently degrading.
    """
    from torchtitan.models.common.attention import FlexAttention

    uses_flex = any(isinstance(m, FlexAttention) for m in model.modules())
    if not uses_flex or backend == "inductor":
        return backend

    if backend != "aot_eager":
        raise ValueError(
            f"Model uses FlexAttention but compile backend {backend!r} is neither "
            f"'inductor' nor 'aot_eager'; the flex region would decompose to eager "
            f"aten ops (no Triton kernel). Use 'inductor' or 'aot_eager'."
        )

    from torch._dynamo.backends.common import aot_autograd
    from torch.fx.passes.regional_inductor import regional_inductor

    global _regional_inductor_enabled
    _regional_inductor_enabled = True

    logger.info("regional_inductor is enabled")
    return aot_autograd(fw_compiler=regional_inductor, bw_compiler=regional_inductor)


def maybe_regional_inductor(
    inductor_configs: dict,
) -> contextlib.AbstractContextManager:
    """Context manager that marks the wrapped region for ``regional_inductor``.

    Returns a null context unless regional inductor is enabled (see
    ``_maybe_regional_inductor_backend``). When enabled, the region is tagged
    with ``compile_with_inductor`` so a non-inductor outer compile lowers just
    this region to inductor with ``inductor_configs``.
    """
    if not _regional_inductor_enabled:
        return contextlib.nullcontext()
    return fx_traceback.annotate(
        {"compile_with_inductor": {"inductor_configs": inductor_configs}}
    )
