# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#2026.7.29 19:47
from dataclasses import dataclass, field

import torch
from torch.distributed.tensor import DTensor

from torchtitan.models.common.rope import _maybe_check_max_pos, CosSinRoPE


class MRoPE(CosSinRoPE):
    """Multi-dimensional RoPE for Qwen3.5 temporal/height/width positions.

    Standard per-layer RoPE: each full-attention layer owns an ``MRoPE`` and
    applies it through ``RoPE.forward`` -> ``_reshape_cache`` ->
    ``apply_rotary_emb``.

    The override handles 3D ``(batch, seq, 3)`` MRoPE positions by building an
    interleaved cos/sin cache. For 2D ``(batch, seq)`` text positions it falls
    back to the plain ``CosSinRoPE`` lookup.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(CosSinRoPE.Config):
        mrope_section: list[int] = field(default_factory=lambda: [24, 20, 20])

    def __init__(self, config: Config):
        if len(config.mrope_section) != 3:
            raise ValueError(
                f"mrope_section must have 3 entries, got {config.mrope_section}."
            )
        super().__init__(config)

    def _reshape_cache(
        self,
        query: torch.Tensor,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Build a query-broadcastable cos/sin cache."""
        if positions is not None and positions.ndim == 3:
            return self._compute_mrope_cache(positions)
        return super()._reshape_cache(query, positions)

    def _compute_mrope_cache(self, position_ids: torch.Tensor) -> torch.Tensor:
        """Build the interleaved cos/sin cache for 3D MRoPE positions.

        Under TP, the persistent RoPE cache is a Replicate DTensor. Every TP
        rank therefore already owns the same local cache and receives the same
        replicated position IDs. The position-resolved cache can be computed
        independently on every rank and wrapped with ``DTensor.from_local``.

        Do not call ``distribute_tensor`` here: it performs a process-group
        broadcast and constructs a C++ ``BroadcastOptions`` object inside the
        compiled TransformerBlock, which TorchDynamo cannot trace.
        """
        cfg = self.config
        assert isinstance(cfg, MRoPE.Config)

        rope_cache = self.cache
        cache_dtensor = rope_cache if isinstance(rope_cache, DTensor) else None

        if cache_dtensor is not None:
            rope_cache = cache_dtensor.to_local()

        pos = (
            position_ids.to_local()
            if isinstance(position_ids, DTensor)
            else position_ids
        )
        pos = pos.to(device=rope_cache.device)

        _maybe_check_max_pos(pos, max_valid_pos=rope_cache.shape[0] - 1)

        head_dim = rope_cache.shape[-1] // 2
        cos_cache = rope_cache[:, :head_dim]
        sin_cache = rope_cache[:, head_dim:]

        # Start from temporal positions for all dimensions, then overwrite the
        # height/width interleaved sections with their own position IDs.
        t_pos = pos[..., 0].long()
        mrope_cos = cos_cache[t_pos]
        mrope_sin = sin_cache[t_pos]

        half = head_dim // 2
        for dim, offset in enumerate((1, 2), start=1):
            length = cfg.mrope_section[dim] * 3
            low = torch.arange(offset, length, 3, device=rope_cache.device)
            col_indices = torch.cat([low, low + half])
            dim_pos = pos[..., dim].long()
            mrope_cos[..., col_indices] = cos_cache[:, col_indices][dim_pos]
            mrope_sin[..., col_indices] = sin_cache[:, col_indices][dim_pos]

        mrope_cache = torch.cat([mrope_cos, mrope_sin], dim=-1).unsqueeze(2)

        if cache_dtensor is not None:
            # Safe because the source cache and positions are replicated across
            # TP, so every rank computes an identical local result. Disabling
            # run_check avoids a collective inside the compiled region.
            return DTensor.from_local(
                mrope_cache,
                cache_dtensor.device_mesh,
                cache_dtensor.placements,
                run_check=False,
            )

        return mrope_cache
