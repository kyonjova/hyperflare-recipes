# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""ERNIE-5.5-Flash-Preview checkpoint with hybrid HCA/DSA attention and bundled DSpark."""

from types import MappingProxyType

from ignition.declaration import (
    Checkpoint,
    RankCosts,
    RankUsage,
    Recipe,
    SamplingDefaults,
    Speculation,
)


def prepare(
    snapshot,
    plan,
    *,
    compile_forwards,
    kernel_cache,
    progress,
    retain_graphs=False,
    core_storage=None,
    ranks=None,
):
    """Prepare the plan's catalog on process-local GPUs or multi-host DGX Sparks over RoCE."""
    from ignition.preparation.rank_group import prepare_group

    from .bootstrap import PreparationConfig, prepare_rank

    speculation = plan.speculation
    drafting = (
        {}
        if speculation is None
        else {
            "speculative_widths": speculation.widths,
            "speculative_token_capacity": plan.speculative_token_capacity,
            "speculative_graph_budget": speculation.graph_budget or 64,
            "speculative_fixed_width": speculation.fixed_width,
            "packed_verification": speculation.packed_verification,
            "draft_prefix_selection": speculation.draft_prefix_selection,
            "draft_head_quantization": speculation.draft_head_quantization,
            "markov_head_quantization": speculation.markov_head_quantization,
        }
    )
    config = PreparationConfig(
        mtp_only=plan.speculation is not None and plan.speculation.mtp_only,
        collective_tuning=plan.collective_tuning,
        collective_ladder=plan.collective_ladder,
        prompt_logprobs=plan.prompt_logprobs,
        compile_workers=plan.compile_workers,
        autotune_effort=plan.autotune_effort,
        tuning_replica=plan.tuning_replica,
        native_dense_activations=plan.native_dense_activations,
        lock_gpu_clocks=plan.lock_gpu_clocks and plan.autotune_effort != "off",
        devices=tuple(range(plan.world_size)) if ranks is None else (0,) * plan.world_size,
        hosts=None if ranks is None else ranks.hosts,
        roce_ring=plan.roce_ring,
        roce_chain=plan.roce_chain,
        compile_decoder=compile_forwards,
        retain_graphs=retain_graphs,
        max_active=plan.concurrency,
        context=plan.recipe.context,
        prefill_chunk_size=plan.prefill_chunk,
        snapshot_budget_bytes=plan.snapshots * plan.costs.state_row_bytes,
        headroom_bytes=plan.headroom_bytes + plan.reserve_bytes,
        kernel_cache=kernel_cache,
        dspark=speculation is not None,
        force_moe_decode_a16=getattr(plan, "force_moe_decode_a16", False),
        expert_scale_compression=getattr(plan, "expert_scale_compression", "csf"),
        stream_overlap=plan.stream_overlap,
        **drafting,
    )
    return prepare_group(
        snapshot,
        config,
        prepare_rank,
        progress=progress,
        core_storage=core_storage,
        ranks=ranks,
        checkpoint=plan.checkpoint,
    )


def usage(prepared):
    return tuple(
        RankUsage(rank["pages"], rank["free_device_bytes"])
        for rank in prepared.memory_report["ranks"]
    )


RECIPE = Recipe(
    name="ernie55",
    draft_head_quantizations=("nvfp4",),
    markov_head_quantizations=("nvfp4",),
    served_model_name="ERNIE-5.5-Flash-Preview",
    # Generation defaults from baidu/ERNIE-5.5-Flash-Preview generation_config.json.
    sampling_defaults=SamplingDefaults(temperature=1.0, top_p=0.95, top_k=-1),
    prepare=prepare,
    usage=usage,
    checkpoints=(
        Checkpoint(
            alias="ernie-5.5-flash",
            repository="baidu/ERNIE-5.5-Flash-Preview",
            revision="ace1100dcde7bf4eb06add6ae11ce4f43a11ddfe",
            architecture="Ernie5_5_MoeForCausalLM",
            stored_bytes=165_882_599_176,
            device_bytes=165_882_599_176,
            draft_bytes=11_400_000_000,
            # Estimate half of backbone MXFP4 expert block-scale bytes saved by CSF.
            scale_compressed_device_bytes={"csf": 161_654_740_744},
        ),
    ),
    context=1048576,
    page_tokens=256,
    concurrency=32,
    max_concurrency=64,
    prefill_chunk=4096,
    speculation=Speculation(
        draft_tokens=7,
        token_capacity=128,
        packed_verification=True,
        draft_prefix_selection="gpu",
        graph_budget=64,
    ),
    max_draft_tokens=7,
    moe_decode_a16_intermediate_width=2048,
    expert_scale_compression="csf",
    # Continuation rows (37 HCA windows + 37 C128 rings + partial page + optional 3 draft windows)
    # and hybrid HCA/DSA pages are replicated per rank.
    ranks=MappingProxyType(
        {
            2: RankCosts(23_684_096, 1_274_304, 14 << 30),
            4: RankCosts(23_684_096, 1_274_304, 24 << 30),
            8: RankCosts(23_684_096, 1_274_304, 32 << 30),
        }
    ),
    plain_ranks=MappingProxyType(
        {
            2: RankCosts(23_459_072, 1_274_304, 12 << 30),
            4: RankCosts(23_459_072, 1_274_304, 22 << 30),
            8: RankCosts(23_459_072, 1_274_304, 30 << 30),
        }
    ),
)
