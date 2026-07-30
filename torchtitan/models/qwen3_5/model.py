# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#version1：2026.7.29 9：57

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F

from fla.ops.gated_delta_rule import (
    chunk_gated_delta_rule as _fla_chunk_gated_delta_rule,
    fused_recurrent_gated_delta_rule as _fla_fused_recurrent_gated_delta_rule,
)
from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.nn.functional import all_gather as autograd_all_gather
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.experimental import local_map
from torch.distributed.tensor.placement_types import Replicate

from torchtitan.models.common import Conv1d, FeedForward, Linear
from torchtitan.models.common.attention import AttentionMasksType, BaseAttention
from torchtitan.models.common.decoder import Decoder
from torchtitan.models.utils import get_moe_model_nparams_and_flops
from torchtitan.protocols.module import Module

from .rope import MRoPE
from .sharding import set_qwen35_sharding_config
from .vision_encoder import Qwen35VisionEncoder



def _cp_all_gather_sequence(
    x: torch.Tensor,
    cp_mesh: DeviceMesh | None,
) -> torch.Tensor:
    """Autograd-aware all-gather of a contiguous sequence shard.

    Qwen3.5's GatedDeltaNet is recurrent along the sequence dimension, so each
    CP rank must reconstruct the original full sequence before evaluating a
    DeltaNet block.  This helper assumes CP input sharding uses contiguous
    chunks (i.e. no head-tail/PTRR load balancer).
    """
    if cp_mesh is None or cp_mesh.size() == 1:
        return x
    parts = autograd_all_gather(x.contiguous(), group=cp_mesh.get_group())
    return torch.cat(tuple(parts), dim=1)


def _cp_local_sequence_slice(
    x: torch.Tensor,
    cp_mesh: DeviceMesh | None,
    local_seq_len: int | None = None,
) -> torch.Tensor:
    """Return this CP rank's contiguous sequence interval."""
    if cp_mesh is None or cp_mesh.size() == 1:
        return x

    cp_size = cp_mesh.size()
    if local_seq_len is None:
        if x.size(1) % cp_size != 0:
            raise ValueError(
                f"Global sequence length {x.size(1)} is not divisible by "
                f"context_parallel_degree={cp_size}."
            )
        local_seq_len = x.size(1) // cp_size

    start = cp_mesh.get_local_rank() * local_seq_len
    if start + local_seq_len > x.size(1):
        raise ValueError(
            f"Cannot extract CP sequence shard: global length={x.size(1)}, "
            f"start={start}, local length={local_seq_len}."
        )
    return x.narrow(1, start, local_seq_len).contiguous()

def _l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    """L2 norm using rsqrt(sum(x²) + eps), not x/max(norm, eps) like F.normalize, to match FLA kernel."""
    return x * torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)


def _torch_native_gated_delta(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
) -> torch.Tensor:
    """Standalone math reference for the gated delta rule recurrence.

    Sequential O(seqlen) loop — use FLA kernels for GPU efficiency.

    Args:
        q, k: (bs, seqlen, n_heads, key_head_dim)
        v: (bs, seqlen, n_heads, value_head_dim)
        g: (bs, seqlen, n_heads) — log-space decay, always negative
        beta: (bs, seqlen, n_heads) — update gate ∈ (0, 1)

    Returns:
        output: (bs, seqlen, n_heads, value_head_dim)
    """
    B, L, H, D_k = q.shape
    D_v = v.shape[-1]
    dtype = q.dtype

    # Upcast to float32 — recurrence accumulates over seqlen steps
    q = _l2norm(q.float(), dim=-1) * (D_k**-0.5)
    k = _l2norm(k.float(), dim=-1)
    v, g, beta = v.float(), g.float(), beta.float()

    output = torch.zeros(B, L, H, D_v, dtype=torch.float32, device=q.device)
    state = torch.zeros(B, H, D_k, D_v, dtype=torch.float32, device=q.device)

    for t in range(L):
        q_t = q[:, t]
        k_t = k[:, t]
        v_t = v[:, t]
        g_t = g[:, t].exp().unsqueeze(-1).unsqueeze(-1)
        b_t = beta[:, t].unsqueeze(-1)

        state = state * g_t
        kv_mem = torch.einsum("bhkv,bhk->bhv", state, k_t)
        delta = (v_t - kv_mem) * b_t
        state = state + torch.einsum("bhk,bhv->bhkv", k_t, delta)
        output[:, t] = torch.einsum("bhkv,bhk->bhv", state, q_t)

    return output.to(dtype)


class SharedExperts(FeedForward):
    """Qwen3.5 shared expert: SwiGLU FFN with a per-token sigmoid gate.

    The output is ``sigmoid(gate(x)) * ffn(x)``. Inherits ``w1/w2/w3`` from
    FeedForward so weight FQNs are unchanged. This gate is specific to
    Qwen3.5; other models use a plain ``FeedForward`` shared expert.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(FeedForward.Config):
        gate: Linear.Config

    def __init__(self, config: Config):
        super().__init__(config)
        self.gate = config.gate.build()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = super().forward(x)
        return torch.sigmoid(self.gate(x)) * out


# -----------------------------------
# 1. OffsetRMSNorm - 全面兼容版本
# -----------------------------------
class OffsetRMSNorm(Module):
    """RMSNorm with offset: ``(1 + weight) * norm(x)``."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        dim: int
        eps: float = 1e-6

    def __init__(self, config: Config):
        super().__init__()
        self.eps = config.eps
        self.weight = nn.Parameter(torch.zeros(config.dim))

    @staticmethod
    def _forward_local(
        x: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
    ) -> torch.Tensor:
        input_dtype = x.dtype
        x_fp32 = x.float()
        variance = x_fp32.pow(2).mean(-1, keepdim=True)
        x_fp32 = x_fp32 * torch.rsqrt(variance + eps)
        return ((1.0 + weight.float()) * x_fp32).to(input_dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.weight

        # PP metadata inference may pass a DTensor activation while FSDP has
        # materialized the norm parameter as a regular Tensor. Compute on the
        # local activation shard, then restore the exact original placement.
        if isinstance(x, DTensor):
            mesh = x.device_mesh
            placements = x.placements
            x_local = x.to_local()
            weight_local = weight.to_local() if isinstance(weight, DTensor) else weight
            out_local = self._forward_local(x_local, weight_local, self.eps)
            return DTensor.from_local(
                out_local,
                mesh,
                placements,
                run_check=False,
            )

        # Symmetric fallback for a DTensor parameter with a plain activation.
        weight_local = weight.to_local() if isinstance(weight, DTensor) else weight
        return self._forward_local(x, weight_local, self.eps)


class RMSNormGated(Module):
    """Gated RMSNorm: ``silu(gate) * weight * norm(x)``."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        dim: int
        eps: float = 1e-6

    def __init__(self, config: Config):
        super().__init__()
        self.eps = config.eps
        self.weight = nn.Parameter(torch.ones(config.dim))

    @staticmethod
    def _forward_local(
        x: torch.Tensor,
        gate: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
    ) -> torch.Tensor:
        input_dtype = x.dtype
        x_fp32 = x.float()
        variance = x_fp32.pow(2).mean(-1, keepdim=True)
        x_fp32 = x_fp32 * torch.rsqrt(variance + eps)
        x_fp32 = (weight.float() * x_fp32).to(input_dtype)
        x_fp32 = x_fp32 * F.silu(gate.float()).to(input_dtype)
        return x_fp32.to(input_dtype)

    def forward(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        weight = self.weight

        # Keep x/gate local during the custom norm, then reconstruct a DTensor
        # with exactly the same mesh and placements as x.
        if isinstance(x, DTensor):
            mesh = x.device_mesh
            placements = x.placements
            x_local = x.to_local()
            gate_local = gate.to_local() if isinstance(gate, DTensor) else gate
            weight_local = weight.to_local() if isinstance(weight, DTensor) else weight
            out_local = self._forward_local(
                x_local,
                gate_local,
                weight_local,
                self.eps,
            )
            return DTensor.from_local(
                out_local,
                mesh,
                placements,
                run_check=False,
            )

        gate_local = gate.to_local() if isinstance(gate, DTensor) else gate
        weight_local = weight.to_local() if isinstance(weight, DTensor) else weight
        return self._forward_local(x, gate_local, weight_local, self.eps)


class GatedDeltaKernel(Module):
    """Stateless dispatch to FLA kernel or pure-torch fallback.

    Provides a module boundary for the sharding code to wrap forward with
    DTensor→local conversion — same pattern as FlexAttention. Handles Q/K
    head expansion for grouped linear attention internally so that
    repeat_interleave runs on local tensors under TP.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        # "fla_chunked": parallel within chunks, fast for training (default)
        # "fla_fused_recurrent": token-by-token, lower memory for long sequences
        # "torch_native": pure-Python reference, for numerical testing only
        backend: Literal[
            "fla_chunked", "fla_fused_recurrent", "torch_native"
        ] = "fla_chunked"

    def __init__(self, config: Config):
        super().__init__()
        self.backend = config.backend

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
    ) -> torch.Tensor:
        expected_ndims = {"q": q.ndim, "k": k.ndim, "v": v.ndim}
        if any(ndim != 4 for ndim in expected_ndims.values()):
            raise ValueError(
                "GatedDeltaKernel expects q/k/v to be 4-D tensors "
                "[B, L, H, D], but got "
                f"q={tuple(q.shape)}, k={tuple(k.shape)}, v={tuple(v.shape)}."
            )
        if g.ndim != 3 or beta.ndim != 3:
            raise ValueError(
                "GatedDeltaKernel expects g/beta to be 3-D tensors "
                "[B, L, H], but got "
                f"g={tuple(g.shape)}, beta={tuple(beta.shape)}."
            )

        # Expand Q/K heads to match V when n_value_heads > n_key_heads
        if q.shape[2] != v.shape[2]:
            assert v.shape[2] % q.shape[2] == 0
            repeat = v.shape[2] // q.shape[2]
            q = q.repeat_interleave(repeat, dim=2).contiguous()
            k = k.repeat_interleave(repeat, dim=2).contiguous()

        if self.backend == "torch_native":
            return _torch_native_gated_delta(q, k, v, g, beta)

        if self.backend == "fla_chunked":
            result = _fla_chunk_gated_delta_rule(
                q.contiguous(),
                k.contiguous(),
                v.contiguous(),
                g.contiguous(),
                beta.contiguous(),
                use_qk_l2norm_in_kernel=True,
            )

        elif self.backend == "fla_fused_recurrent":
            result = _fla_fused_recurrent_gated_delta_rule(
                q.contiguous(),
                k.contiguous(),
                v.contiguous(),
                g.contiguous(),
                beta=beta.contiguous(),
                use_qk_l2norm_in_kernel=True,
            )
        else:
            raise ValueError(
                f"Unknown fla_backend '{self.backend}'. "
                "Valid: 'fla_chunked', 'fla_fused_recurrent', 'torch_native'."
            )

        # FLA kernels return (output, final_state); we only need output
        out = result[0]
        
        # 🔧 关键修改：FLA 内核内部会使用 float32 进行累加计算以保证数值稳定，
        # 返回的输出通常是 float32 类型。这会导致后续网络层（如 norm 和 out_proj）
        # 的计算和梯度都变成 float32，进而触发 FSDP 的 reduce-scatter 梯度类型不一致错误。
        # 必须强制转换回输入时的数据类型（如 bfloat16）。
        if out.dtype != q.dtype:
            out = out.to(q.dtype)
            
        return out

class GatedDeltaNet(Module):
    """Gated DeltaNet linear attention."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        key_head_dim: int
        value_head_dim: int
        conv_kernel_size: int = 4
        in_proj_q: Linear.Config
        in_proj_k: Linear.Config
        in_proj_v: Linear.Config
        in_proj_z: Linear.Config
        in_proj_a: Linear.Config
        in_proj_b: Linear.Config
        conv_q: Conv1d.Config
        conv_k: Conv1d.Config
        conv_v: Conv1d.Config
        kernel: GatedDeltaKernel.Config
        norm: RMSNormGated.Config
        out_proj: Linear.Config

    def __init__(self, config: Config):
        super().__init__()
        self.key_head_dim = config.key_head_dim
        self.value_head_dim = config.value_head_dim
        self.conv_kernel_size = config.conv_kernel_size

        value_dim = config.in_proj_v.out_features

        self.in_proj_q = config.in_proj_q.build()
        self.in_proj_k = config.in_proj_k.build()
        self.in_proj_v = config.in_proj_v.build()
        self.in_proj_z = config.in_proj_z.build()
        self.in_proj_a = config.in_proj_a.build()
        self.in_proj_b = config.in_proj_b.build()

        self.conv_q = config.conv_q.build()
        self.conv_k = config.conv_k.build()
        self.conv_v = config.conv_v.build()

        n_value_heads = value_dim // config.value_head_dim
        self.A_log = nn.Parameter(torch.zeros(n_value_heads))
        self.dt_bias = nn.Parameter(torch.zeros(n_value_heads))

        self.kernel = config.kernel.build()
        self.norm = config.norm.build()
        self.out_proj = config.out_proj.build()


    def _causal_conv(self, x: torch.Tensor, conv: nn.Module) -> torch.Tensor:
        if isinstance(x, DTensor):
            x_plc = x.placements
            w = conv.weight
            w_plc = w.placements if isinstance(w, DTensor) else [Replicate()] * w.ndim
        
            def _conv(x_local: torch.Tensor, w_local: torch.Tensor) -> torch.Tensor:
                x_local = x_local.transpose(1, 2)
                x_local = F.pad(x_local, [self.conv_kernel_size - 1, 0])
                out = F.conv1d(
                    x_local,
                    w_local,
                    None,
                    conv.stride,
                    conv.padding,
                    conv.dilation,
                    w_local.size(0),
                )
                return out.transpose(1, 2)
        
            conv_dt = local_map(
                _conv,
                out_placements=(x_plc,),
                in_placements=(x_plc, w_plc),
                in_grad_placements=(x_plc, w_plc),
                device_mesh=x.device_mesh,
            )
            if not isinstance(w, DTensor):
                w = DTensor.from_local(w, x.device_mesh, w_plc, run_check=False)
            x = conv_dt(x, w)
            return F.silu(x)
        else:
            x = F.pad(x.transpose(1, 2), [self.conv_kernel_size - 1, 0])
            x = conv(x)
            return F.silu(x).transpose(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bs, seqlen, _ = x.shape

        # Call the configured modules directly. Module.parallelize() has wrapped
        # their forward methods with the placements declared in sharding.py.
        # Calling F.linear()/to_local() manually bypasses those wrappers.
        xq = self._causal_conv(self.in_proj_q(x), self.conv_q)
        xk = self._causal_conv(self.in_proj_k(x), self.conv_k)
        xv = self._causal_conv(self.in_proj_v(x), self.conv_v)
        xz = self.in_proj_z(x)
        xa = self.in_proj_a(x)
        xb = self.in_proj_b(x)

        # FLA expects explicit head dimensions.
        xq = xq.reshape(bs, seqlen, -1, self.key_head_dim)
        xk = xk.reshape(bs, seqlen, -1, self.key_head_dim)
        xv = xv.reshape(bs, seqlen, -1, self.value_head_dim)
        xz = xz.reshape(bs, seqlen, -1, self.value_head_dim)

        # Per-value-head decay and update gates.
        g = -torch.exp(self.A_log.float()) * F.softplus(
            xa.float() + self.dt_bias.float()
        )
        g = g.to(xa.dtype)
        beta = torch.sigmoid(xb)

        # GatedDeltaKernel's ShardingConfig/local_map converts DTensors to local
        # tensors for FLA and restores the DTensor output automatically.
        output = self.kernel(xq, xk, xv, g, beta)
        output = self.norm(output, xz)

        output = output.reshape(bs, seqlen, -1)
        return self.out_proj(output)




class Qwen35Attention(BaseAttention):
    """Full attention with output gating and partial RoPE for Qwen3.5.

    Differences from GQAttention:
    - wq is 2x wider: produces both query and sigmoid gate
    - Partial RoPE: only first ``rotary_dim`` elements get RoPE
    - Output gating: ``attn_output * sigmoid(gate)`` before ``wo``
    - QK norm uses OffsetRMSNorm

    Uses separate ``wq``/``wk``/``wv`` instead of the common fused ``qkv_linear``
    (so this subclasses ``BaseAttention``, not ``GQAttention``): the 2x-wide,
    gated ``wq`` doesn't fit a fused QKV projection that TP-shards by head.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(BaseAttention.Config):
        n_heads: int
        n_kv_heads: int
        head_dim: int
        rotary_dim: int
        rope: MRoPE.Config
        wq: Linear.Config
        wk: Linear.Config
        wv: Linear.Config
        wo: Linear.Config
        q_norm: OffsetRMSNorm.Config
        k_norm: OffsetRMSNorm.Config
        inner_attention: Module.Config

    def __init__(self, config: Config):
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.head_dim
        self.rotary_dim = config.rotary_dim
        self.enable_gqa = self.n_heads > self.n_kv_heads

        self.wq = config.wq.build()
        self.wk = config.wk.build()
        self.wv = config.wv.build()
        self.wo = config.wo.build()

        self.rope = config.rope.build()

        self.q_norm = config.q_norm.build()
        self.k_norm = config.k_norm.build()

        self.scaling = self.head_dim**-0.5

        self.inner_attention = config.inner_attention.build()

    def forward(
        self,
        x: torch.Tensor,
        attention_masks: AttentionMasksType | None,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        bs, seqlen, _ = x.shape

        # wq is 2x wider: produces query + gate
        xq_gate = self.wq(x).view(bs, seqlen, -1, self.head_dim * 2)
        xq, gate = xq_gate.chunk(2, dim=-1)
        xk = self.wk(x).view(bs, seqlen, -1, self.head_dim)
        xv = self.wv(x).view(bs, seqlen, -1, self.head_dim)

        # QK norm (before RoPE)
        xq = self.q_norm(xq)
        xk = self.k_norm(xk)

        # Partial RoPE: only first rotary_dim elements get positional encoding
        assert self.rotary_dim <= self.head_dim
        xq_rot, xq_pass = xq[..., : self.rotary_dim], xq[..., self.rotary_dim :]
        xk_rot, xk_pass = xk[..., : self.rotary_dim], xk[..., self.rotary_dim :]
        xq_rot, xk_rot = self.rope(xq_rot, xk_rot, positions)
        xq = torch.cat([xq_rot, xq_pass], dim=-1)
        xk = torch.cat([xk_rot, xk_pass], dim=-1)

        output = self.inner_attention(
            xq,
            xk,
            xv,
            attention_masks=attention_masks,
            scale=self.scaling,
            enable_gqa=self.enable_gqa,
        ).contiguous()

        # Output gating
        output = output * torch.sigmoid(gate)
        output = output.view(bs, seqlen, -1)
        return self.wo(output)


class Qwen35TransformerBlock(Module):
    """Hybrid transformer block for Qwen3.5."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        attention: Qwen35Attention.Config | None = None
        delta_net: GatedDeltaNet.Config | None = None
        feed_forward: Module.Config | None = None
        moe: Module.Config | None = None
        attention_norm: OffsetRMSNorm.Config
        ffn_norm: OffsetRMSNorm.Config

    def __init__(self, config: Config):
        super().__init__()
        self.full_attn = config.attention is not None

        if self.full_attn:
            self.attn = config.attention.build()  # pyrefly: ignore [missing-attribute]
        else:
            assert config.delta_net is not None
            self.attn = config.delta_net.build()

        self.moe_enabled = config.moe is not None
        if self.moe_enabled:
            # pyrefly: ignore [missing-attribute]
            self.moe = config.moe.build()
        else:
            assert config.feed_forward is not None
            self.feed_forward = config.feed_forward.build()

        self.attention_norm = config.attention_norm.build()
        self.ffn_norm = config.ffn_norm.build()
    
    def forward(
        self,
        x: torch.Tensor,
        attention_masks: AttentionMasksType | None,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = self.attention_norm(x)
        if self.full_attn:
            h = self.attn(h, attention_masks, positions)
        else:
            h = self.attn(h)
        x = x + h

        h = self.ffn_norm(x)
        if self.moe_enabled:
            x = x + self.moe(h)
        else:
            x = x + self.feed_forward(h)
        return x



class Qwen35Model(Decoder):
    """Qwen3.5: Multimodal model with hybrid attention.

    Combines a hybrid decoder (GatedDeltaNet linear attention + full
    attention with output gating and partial RoPE) with a Vision
    Transformer encoder for multimodal understanding.

    Key architectural features:
    - Hybrid attention: 75% GatedDeltaNet (linear) + 25% full attention
    - Output gating on full attention: ``attn_out * sigmoid(gate)``
    - Partial RoPE: only first ``rotary_dim`` elements get positional encoding
    - OffsetRMSNorm: ``(1 + weight) * norm(x)`` with zero-init weight
    - MRoPE: 3D (temporal/height/width) position IDs for multimodal batches;
      text batches use the plain 1D positions
    - MoE variant: routed experts + shared expert with sigmoid gate

    MRoPE positions (``mrope_positions``, shape ``(batch, seq, 3)``) are built by
    the dataloader and forwarded to every pipeline stage, so RoPE stays consistent
    across stages even though the raw vision inputs (``pixel_values``/``grid_thw``)
    only reach the first stage. Text batches carry no ``mrope_positions`` and use
    the 2D ``positions`` instead.

    Forward pass flow::

        forward(tokens, pixel_values, grid_thw, mrope_positions, ...)
          │
          ├─ _prepare_multimodal_embeds
          │    ├─ tok_embeddings(tokens)              → text embeddings
          │    ├─ _get_vision_embeds(pixel_values)     → vision embeddings
          │    │    └─ vision_encoder(pixel_values)     → merge patches
          │    ├─ _get_vision_positions             → locate vision regions
          │    └─ _scatter_vision_embeds                → scatter into text sequence
          │
          └─ transformer layers (hybrid), each given (mrope_positions or positions)
               └─ for each layer:
                    ├─ full attention (every Nth):  QK-norm → partial RoPE → SDPA → gate
                    │    (the layer's MRoPE builds the cos/sin cache from positions)
                    └─ GatedDeltaNet (others):      Conv1d → gated delta rule → gated norm
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Decoder.Config):
        vision_encoder: Qwen35VisionEncoder.Config

        def update_from_config(
            self,
            *,
            config,
            **kwargs,
        ) -> None:
            Decoder.Config.update_from_config(self, config=config, **kwargs)
            parallelism = config.parallelism

            tp = parallelism.tensor_parallel_degree
            if tp > 1:
                dn_cfg = next(
                    (l.delta_net for l in self.layers if l.delta_net is not None),
                    None,
                )
                if dn_cfg is not None:
                    n_key_heads = dn_cfg.in_proj_q.out_features // dn_cfg.key_head_dim
                    n_value_heads = (
                        dn_cfg.in_proj_v.out_features // dn_cfg.value_head_dim
                    )
                    if n_key_heads % tp != 0 or n_value_heads % tp != 0:
                        raise ValueError(
                            f"tensor_parallel_degree ({tp}) must divide "
                            f"n_key_heads ({n_key_heads}) and "
                            f"n_value_heads ({n_value_heads})."
                        )

            set_qwen35_sharding_config(
                self,
                enable_ep=parallelism.expert_parallel_degree > 1,
            )

        def get_nparams_and_flops(
            self, model: nn.Module, seq_len: int
        ) -> tuple[int, int]:
            attn_cfg = self.first_attention
            # pyrefly: ignore [missing-attribute]
            n_heads = attn_cfg.n_heads
            # pyrefly: ignore [missing-attribute]
            head_dim = attn_cfg.head_dim
            return get_moe_model_nparams_and_flops(
                self,
                model,
                n_heads,
                2 * head_dim,
                seq_len,
            )

    def __init__(self, config: Config):
        super().__init__(config)

        self.vision_encoder = config.vision_encoder.build()
        self.spatial_merge_size = config.vision_encoder.spatial_merge_size
        self._cp_mesh: DeviceMesh | None = None

    def enable_context_parallel(self, cp_mesh: DeviceMesh) -> None:
        """Enable the experimental contiguous-sequence CP execution path."""
        self._cp_mesh = cp_mesh
    '''
    def _get_vision_positions(
        self,
        tokens: torch.Tensor,
        num_tokens_per_item: torch.Tensor,
        vision_token_id: int,
    ) -> list[tuple[int, int, int, int]]:
        """Compute (item_idx, sample_idx, vision_start, n_tokens) for each vision item.

        Finds where each contiguous run of vision placeholder tokens starts
        in the text sequence.

        Args:
            tokens: Token IDs (batch, seq_len)
            num_tokens_per_item: (num_items,) actual tokens per vision item
            vision_token_id: Placeholder token ID

        Returns:
            List of (item_idx, sample_idx, vision_start, n_tokens) tuples
        """
        vision_mask = tokens == vision_token_id
        flat_mask = vision_mask.view(-1)
        prev_mask = torch.cat(
            [torch.zeros(1, dtype=torch.bool, device=flat_mask.device), flat_mask[:-1]]
        )
        region_starts = torch.where(flat_mask & ~prev_mask)[0]
        seq_len = tokens.shape[1]

        positions = []
        for i in range(num_tokens_per_item.shape[0]):
            start = int(region_starts[i].item())
            n_tokens = int(num_tokens_per_item[i].item())
            positions.append((i, start // seq_len, start % seq_len, n_tokens))
        return positions
    '''
    def _get_vision_positions(
        self,
        tokens: torch.Tensor,
        num_tokens_per_item: torch.Tensor,
        vision_token_id: int,
    ) -> list[tuple[int, int, int, int]]:
        """Compute (item_idx, sample_idx, vision_start, n_tokens) for each vision item.

        Finds where each contiguous run of vision placeholder tokens starts
        in the text sequence.

        Args:
            tokens: Token IDs (batch, seq_len)
            num_tokens_per_item: (num_items,) actual tokens per vision item
            vision_token_id: Placeholder token ID

        Returns:
            List of (item_idx, sample_idx, vision_start, n_tokens) tuples
        """
        vision_mask = tokens == vision_token_id
        flat_mask = vision_mask.view(-1)
        prev_mask = torch.cat(
            [torch.zeros(1, dtype=torch.bool, device=flat_mask.device), flat_mask[:-1]]
        )
        region_starts = torch.where(flat_mask & ~prev_mask)[0]
    
        # 🔧 修改：检查region_starts是否为空
        if len(region_starts) == 0:
            return []  # 无视觉占位符，返回空列表
    
        seq_len = tokens.shape[1]
        positions = []
        for i in range(num_tokens_per_item.shape[0]):
            if i < len(region_starts):  # 🔧 修改：防止索引越界
                start = int(region_starts[i].item())
                n_tokens = int(num_tokens_per_item[i].item())
                positions.append((i, start // seq_len, start % seq_len, n_tokens))
        return positions

    def _get_vision_embeds(
        self,
        pixel_values: torch.Tensor,
        *,
        grid_thw: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run vision encoder and return padded embeddings with token counts.

        Args:
            pixel_values: Padded patches (num_items, max_num_patch, patch_dim)
            grid_thw: Grid dimensions (num_items, 3) for [t, h, w]

        Returns:
            merged_embeds: (num_items, max_tokens, dim) padded vision embeddings
            num_tokens_per_item: (num_items,) actual token count per item
        """
        pixel_values = pixel_values.to(self.vision_encoder.patch_embed.weight.dtype)
        merged_embeds = self.vision_encoder(
            pixel_values, grid_thw=grid_thw
        ).clone()

        merge_unit = self.vision_encoder.spatial_merge_unit
        num_tokens_per_item = grid_thw.prod(-1) // merge_unit

        return merged_embeds, num_tokens_per_item

    def _scatter_vision_embeds(
        self,
        inputs_embeds: torch.Tensor,
        *,
        merged_embeds: torch.Tensor,
        vision_positions: list[tuple[int, int, int, int]],
    ) -> torch.Tensor:
        """Scatter vision embeddings into text embeddings at placeholder positions.

        Copies directly from the padded vision encoder output into the text
        sequence.

        Args:
            inputs_embeds: Text embeddings (batch, seq_len, dim)
            merged_embeds: Padded vision embeddings (num_items, max_tokens, dim)
            vision_positions: List of (item_idx, sample_idx, vision_start, n_tokens)

        Returns:
            Updated embeddings
        """
        for item_idx, sample_idx, vision_start, n_tokens in vision_positions:
            inputs_embeds[
                sample_idx, vision_start : vision_start + n_tokens, :
            ] = merged_embeds[item_idx, :n_tokens, :]
        return inputs_embeds

    def _prepare_multimodal_embeds(
        self,
        tokens: torch.Tensor,
        *,
        pixel_values: torch.Tensor | None,
        pixel_values_videos: torch.Tensor | None,
        grid_thw: torch.Tensor | None,
        grid_thw_videos: torch.Tensor | None,
        special_tokens: dict[str, int],
    ) -> torch.Tensor:
        """Embed tokens, run vision encoder, scatter vision into text.

        Args:
            tokens: Input token IDs (batch_size, seq_len)
            pixel_values: Image patches or None
            pixel_values_videos: Video patches or None
            grid_thw: Grid dimensions for images or None
            grid_thw_videos: Grid dimensions for videos or None
            special_tokens: Special token definitions

        Returns:
            (batch, seq_len, dim) embeddings with vision tokens scattered in
        """
        image_token_id = special_tokens["image_id"]
        video_token_id = special_tokens["video_id"]

        inputs_embeds = (
            self.tok_embeddings(tokens) if self.tok_embeddings is not None else tokens
        )

        if pixel_values is not None and grid_thw is not None:
            merged_embeds, num_tokens = self._get_vision_embeds(
                pixel_values, grid_thw=grid_thw
            )
            image_positions = self._get_vision_positions(
                tokens, num_tokens, image_token_id
            )
            if image_positions:
                inputs_embeds = self._scatter_vision_embeds(
                    inputs_embeds,
                    merged_embeds=merged_embeds,
                    vision_positions=image_positions,
                )

        if pixel_values_videos is not None and grid_thw_videos is not None:
            merged_embeds, num_tokens = self._get_vision_embeds(
                pixel_values_videos, grid_thw=grid_thw_videos
            )
            video_positions = self._get_vision_positions(
                tokens, num_tokens, video_token_id
            )
            if video_positions:
                inputs_embeds = self._scatter_vision_embeds(
                    inputs_embeds,
                    merged_embeds=merged_embeds,
                    vision_positions=video_positions,
                )

        return inputs_embeds

    def forward(  # pyrefly: ignore [bad-override]
        self,
        tokens: torch.Tensor,
        *,
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.Tensor | None = None,
        grid_thw: torch.Tensor | None = None,
        grid_thw_videos: torch.Tensor | None = None,
        attention_masks: AttentionMasksType | None = None,
        positions: torch.Tensor | None = None,
        mrope_positions: torch.Tensor | None = None,
        special_tokens: dict[str, int] | None = None,
        loss_kwargs: dict | None = None,
    ) -> torch.Tensor:
        del loss_kwargs  # consumed by the pipeline loss function, not the model

        cp_mesh = self._cp_mesh
        local_seq_len = tokens.size(1)

        if self.tok_embeddings is not None:
            # The trainer CP-shards raw tokens before model.forward(). Vision
            # placeholders, image/video embeddings and MRoPE metadata are global,
            # so rebuild the full token sequence before multimodal scatter, then
            # retain only this rank's contiguous sequence interval.
            embedding_tokens = _cp_all_gather_sequence(tokens, cp_mesh)
            x = self._prepare_multimodal_embeds(
                embedding_tokens,
                pixel_values=pixel_values,
                pixel_values_videos=pixel_values_videos,
                grid_thw=grid_thw,
                grid_thw_videos=grid_thw_videos,
                special_tokens=special_tokens,
            )
            x = _cp_local_sequence_slice(x, cp_mesh, local_seq_len)
        else:
            # Non-first PP stages receive hidden states in the `tokens` slot.
            x = tokens

        # Only forward-SPMD parallelization (TP/EP/spmd_types) should turn
        # activations into DTensors. FSDP2 also stores sharded parameters as
        # DTensors between forwards, but its pre-forward hook materializes plain
        # Tensor parameters for computation. Treating an FSDP storage parameter
        # as an activation mesh would produce DTensor activations with Tensor
        # weights and fail in Linear/Conv operators.
        mesh_param = None
        if (
            cp_mesh is None
            and self._parallelized
            and not isinstance(x, DTensor)
        ):
            try:
                mesh_param = next(self.parameters())
            except StopIteration:
                mesh_param = None
            if isinstance(mesh_param, DTensor):
                mesh = mesh_param.device_mesh
                placements = [Replicate()] * mesh.ndim
                x = DTensor.from_local(x, mesh, placements, run_check=False)

        rope_positions = mrope_positions if mrope_positions is not None else positions
        assert rope_positions is not None

        # prepare_context_parallel_input() shards `positions`, but Qwen3.5's
        # three-axis `mrope_positions` may still arrive at global length.
        if cp_mesh is not None and rope_positions.size(1) != x.size(1):
            expected_global = x.size(1) * cp_mesh.size()
            if rope_positions.size(1) != expected_global:
                raise ValueError(
                    f"RoPE sequence length {rope_positions.size(1)} does not match "
                    f"local CP length {x.size(1)} or global length {expected_global}."
                )
            rope_positions = _cp_local_sequence_slice(
                rope_positions, cp_mesh, x.size(1)
            )

        for layer in self.layers.values():
            x = layer(x, attention_masks, rope_positions)
            if (
                cp_mesh is None
                and self._parallelized
                and not isinstance(x, DTensor)
                and isinstance(mesh_param, DTensor)
            ):
                mesh = mesh_param.device_mesh
                placements = [Replicate()] * mesh.ndim
                x = DTensor.from_local(x, mesh, placements, run_check=False)

        x = self.norm(x) if self.norm is not None else x

        if self._skip_lm_head:
            if cp_mesh is None and isinstance(x, DTensor):
                x = x.to_local() if x.placements[0].is_replicate() else x.full_tensor()
            return x

        if self.lm_head is not None:
            out = self.lm_head(x)
            if cp_mesh is None and isinstance(out, DTensor):
                out = out.full_tensor()
            return out

        return x
