# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#2026.7.29 20:03
"""
Parallelization utilities for Qwen3.5.

This module applies PT-D parallelisms and various training techniques
(activation checkpointing, compile, FSDP) to the Qwen3.5 model.
"""

import torch
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.nn.functional import all_gather as autograd_all_gather
from torch.distributed.tensor import DTensor
from torch.distributed._composable.fsdp import fully_shard
from torch.distributed.fsdp import MixedPrecisionPolicy

from torchtitan.config import (
    CompileConfig,
    ParallelismConfig,
    TORCH_DTYPE_MAP,
    TrainingConfig,
)

from torchtitan.distributed import ParallelDims
from torchtitan.distributed.activation_checkpoint import ActivationCheckpointingConfig
from torchtitan.distributed.compile import apply_compile
from torchtitan.distributed.context_parallel import apply_cp_to_forward
from torchtitan.distributed.fsdp import (
    apply_fsdp_to_decoder,
    get_fsdp_reshard_after_forward_policy,
)
from torchtitan.distributed.tensor_parallel import maybe_enable_async_tp
from torchtitan.tools.logging import logger


def _apply_cp_to_deltanet(
    deltanet_modules: list[nn.Module],
    cp_mesh: DeviceMesh,
) -> None:
    """Apply a correctness-first CP wrapper to recurrent DeltaNet blocks.

    Each rank receives a contiguous sequence shard. The wrapper performs an
    autograd-aware all-gather, evaluates the recurrent block on the original
    full sequence, and returns only the caller rank's interval. This is
    mathematically correct but duplicates DeltaNet compute and therefore is an
    experimental compatibility path, not an optimized ring/recurrent CP kernel.
    """
    cp_group = cp_mesh.get_group()
    cp_rank = cp_mesh.get_local_rank()

    for module in deltanet_modules:
        original_forward = module.forward

        def _make_cp_forward(orig_fn, group, rank):
            def cp_forward(x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
                if isinstance(x, DTensor):
                    raise RuntimeError(
                        "Experimental Qwen3.5 DeltaNet CP expects local tensors. "
                        "Use --parallelism.spmd_backend default and TP=1."
                    )
                local_seq_len = x.size(1)
                parts = autograd_all_gather(x.contiguous(), group=group)
                full_x = torch.cat(tuple(parts), dim=1)
                full_out = orig_fn(full_x, *args, **kwargs)
                start = rank * local_seq_len
                return full_out.narrow(1, start, local_seq_len).contiguous()

            return cp_forward

        module.forward = _make_cp_forward(original_forward, cp_group, cp_rank)

    logger.warning(
        "Enabled experimental Qwen3.5 DeltaNet CP: every DeltaNet block "
        "all-gathers the full sequence and duplicates its compute."
    )


def _apply_fsdp_to_vision_encoder(
    vision_encoder: nn.Module,
    dp_mesh,
    param_dtype: torch.dtype,
    reduce_dtype: torch.dtype,
    reshard_after_forward_policy: str = "default",
    pp_enabled: bool = False,
):
    """FSDP the vision encoder as a single unit.

    One AllGather for all vision params is more efficient than per-layer
    sharding — the vision encoder is small relative to the decoder.
    Must be called before apply_fsdp on the decoder.
    """
    mp_policy = MixedPrecisionPolicy(param_dtype=param_dtype, reduce_dtype=reduce_dtype)
    reshard_after_forward = get_fsdp_reshard_after_forward_policy(
        reshard_after_forward_policy, pp_enabled=pp_enabled
    )
    fully_shard(
        vision_encoder,
        mesh=dp_mesh,
        mp_policy=mp_policy,
        reshard_after_forward=reshard_after_forward,
    )


def parallelize_qwen3_5(
    model: nn.Module,
    *,
    parallel_dims: ParallelDims,
    training: TrainingConfig,
    parallelism: ParallelismConfig,
    compile_config: CompileConfig,
    ac_config: ActivationCheckpointingConfig,
    dump_folder: str,
):
    """
    Apply tensor parallelism, activation checkpointing, torch.compile, and data
    parallelism to the Qwen3.5 model.

    NOTE: The passed-in model preferably should be on meta device. Otherwise,
    the model must fit on GPU or CPU memory.
    """
    if parallelism.spmd_backend == "full_dtensor":
        raise NotImplementedError("full_dtensor is not supported yet.")

    model_compile_enabled = (
        compile_config.enable and "model" in compile_config.components
    )

    if parallel_dims.cp_enabled:
        if parallel_dims.pp_enabled:
            raise NotImplementedError(
                "The experimental Qwen3.5 CP path supports FSDP+CP only; "
                "PP+CP is not enabled. Set pipeline_parallel_degree=1."
            )
        if model_compile_enabled:
            raise NotImplementedError(
                "The experimental Qwen3.5 CP path currently requires "
                "compile.enable=false."
            )
        if parallel_dims.tp_enabled or parallel_dims.ep_enabled:
            raise NotImplementedError(
                "The experimental Qwen3.5 CP path currently requires TP=1 and EP=1."
            )
        if parallelism.spmd_backend != "default":
            raise NotImplementedError(
                "The experimental Qwen3.5 CP path requires "
                "--parallelism.spmd_backend default."
            )

        # DeltaNet is causal/recurrent and requires the original token order.
        # Head-tail or PTRR reordering cannot be reconstructed by a simple rank-
        # ordered all-gather, so force contiguous CP shards. This mutates the same
        # config object later consumed by Trainer.post_dataloading_process().
        if parallelism.context_parallel_load_balancer is not None:
            logger.warning(
                "Qwen3.5 DeltaNet CP requires contiguous sequence shards; "
                "disabling context_parallel_load_balancer=%r.",
                parallelism.context_parallel_load_balancer,
            )
            parallelism.context_parallel_load_balancer = None

        cp_mesh = parallel_dims.get_mesh("cp")
        full_attention_modules: list[nn.Module] = []
        deltanet_modules: list[nn.Module] = []
        for block in model.layers.values():
            if getattr(block, "full_attn", False):
                full_attention_modules.append(block.attn.inner_attention)
            else:
                deltanet_modules.append(block.attn)

        if full_attention_modules:
            # Flex/SDPA CP for the 25% full-attention blocks.
            apply_cp_to_forward(full_attention_modules, cp_mesh)
        _apply_cp_to_deltanet(deltanet_modules, cp_mesh)

        if not hasattr(model, "enable_context_parallel"):
            raise RuntimeError(
                "Qwen35Model.enable_context_parallel() is missing. Replace "
                "model.py and parallelize.py as a matched pair."
            )
        model.enable_context_parallel(cp_mesh)

    # Qwen3.5 declares DTensor placements for norms, DeltaNet kernels and
    # projections through the spmd_types sharding configuration. These
    # placements must also be installed when TP=1 and EP=1; PP shape inference
    # may otherwise produce DTensor activations while parameters remain ordinary
    # torch.Tensor objects, causing mixed Tensor/DTensor operator failures.
    if (
        parallelism.spmd_backend == "spmd_types"
        or parallel_dims.tp_enabled
        or parallel_dims.ep_enabled
    ):
        if parallelism.enable_async_tensor_parallel and not model_compile_enabled:
            raise RuntimeError("Async TP requires torch.compile")

        # pyrefly: ignore [not-callable]
        model.parallelize(parallel_dims)

    if parallel_dims.tp_enabled:
        maybe_enable_async_tp(parallelism, compile_config, parallel_dims.get_mesh("tp"))

    if ac_config is not None:
        ac_policy = ac_config.build(dump_folder=dump_folder)
        ac_policy.apply(model)
        if model.vision_encoder is not None:
            ac_policy.apply(model.vision_encoder)

    if model_compile_enabled:
        # FLA's chunk_gated_delta_rule is wrapped with torch.compiler.disable in
        # the installed FLA package. A fullgraph compile of a checkpointed
        # DeltaNet block therefore fails. Compile only Qwen3.5 Full Attention
        # blocks and keep DeltaNet/FLA blocks in eager mode.
        apply_compile(
            model,
            compile_config,
            layer_filter=lambda _layer_id, block: bool(
                getattr(block, "full_attn", False)
            ),
        )

        if model.vision_encoder is not None:
            # Vision blocks do not use the disabled FLA DeltaNet kernel, so they
            # can still be compiled normally.
            # pyrefly: ignore [bad-argument-type]
            apply_compile(model.vision_encoder, compile_config)

    dp_mesh_names = (
        ["dp_replicate", "fsdp"] if parallel_dims.dp_replicate_enabled else ["fsdp"]
    )
    dp_mesh = parallel_dims.get_mesh(dp_mesh_names)

    if model.vision_encoder is not None:
        _apply_fsdp_to_vision_encoder(
            model.vision_encoder,  # pyrefly: ignore [bad-argument-type]
            dp_mesh,
            param_dtype=TORCH_DTYPE_MAP[training.mixed_precision_param],
            reduce_dtype=TORCH_DTYPE_MAP[training.mixed_precision_reduce],
            reshard_after_forward_policy=parallelism.fsdp_reshard_after_forward,
            pp_enabled=parallel_dims.pp_enabled,
        )

    edp_mesh = None
    if parallel_dims.ep_enabled:
        edp_mesh_names = (
            ["dp_replicate", "efsdp"]
            if parallel_dims.dp_replicate_enabled
            else ["efsdp"]
        )
        edp_mesh = parallel_dims.get_optional_mesh(edp_mesh_names)

    apply_fsdp_to_decoder(
        model,  # pyrefly: ignore [bad-argument-type]
        dp_mesh,
        param_dtype=TORCH_DTYPE_MAP[training.mixed_precision_param],
        reduce_dtype=TORCH_DTYPE_MAP[training.mixed_precision_reduce],
        pp_enabled=parallel_dims.pp_enabled,
        cpu_offload=training.enable_cpu_offload,
        reshard_after_forward_policy=parallelism.fsdp_reshard_after_forward,
        ep_degree=parallel_dims.ep,
        edp_mesh=edp_mesh,
    )

    return model


def pipeline_qwen3_5(
    model: nn.Module,
    *,
    parallel_dims: ParallelDims,
    parallelism: ParallelismConfig,
    model_config,
    **kwargs,
):
    """PP wrapper that assigns vision_encoder to the first pipeline stage.

    Delegates to ``pipeline_llm`` after injecting ``vision_encoder`` into
    the first stage's FQN list (the auto-generated LLM split doesn't know
    about vision encoder modules).
    """
    import dataclasses

    from torchtitan.distributed.pipeline_parallel import (
        _generate_llm_fqn_per_model_part,
        _get_pipeline_metadata,
        pipeline_llm,
    )

    if parallelism.module_fqns_per_model_part is None:
        (
            num_virtual_stages,
            num_layers,
            input_weight,
            output_weight,
        ) = _get_pipeline_metadata(parallel_dims, parallelism, model_config)
        fqn_per_part = _generate_llm_fqn_per_model_part(
            num_virtual_stages, num_layers, input_weight, output_weight
        )
        # Vision encoder lives on the first stage alongside tok_embeddings. This
        # adds load to stage 0 that the auto split doesn't model (input_weight
        # only accounts for tok_embeddings); for a heavy vision encoder, bump
        # parallelism.pipeline_parallel_first_stage_less_layers to rebalance.
        if hasattr(model, "vision_encoder") and model.vision_encoder is not None:
            fqn_per_part[0].insert(0, "vision_encoder")
        parallelism = dataclasses.replace(
            parallelism, module_fqns_per_model_part=fqn_per_part
        )

    return pipeline_llm(
        model,
        parallel_dims=parallel_dims,
        parallelism=parallelism,
        model_config=model_config,
        **kwargs,
    )
