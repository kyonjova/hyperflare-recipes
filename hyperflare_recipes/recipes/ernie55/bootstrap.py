# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Prepare ERNIE-5.5-Flash-Preview hybrid HCA/DSA target and bundled DSpark graph catalogs."""

import contextlib
import hashlib
import json
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path

import torch

from ignition.components.capacity import (
    batch_capacities,
    packed_speculative_shapes,
    prefill_capacities,
    proposal_shapes,
    request_capacity,
    serving_modes,
    speculative_shapes,
    speculative_token_capacity,
)
from ignition.components.completion import VECTOR_TOKENS_FIELD, CompletionOutputs
from ignition.components.sampling import AdmissionBuffers, DeviceRequestState
from ignition.components.state_transfer import StateCommands
from ignition.preparation.bootstrap import BasePreparationConfig
from ignition.preparation.capture import capture_forwards
from ignition.preparation.checkpoint import inspect_indexed_checkpoint
from ignition.preparation.collectives import (
    CollectivePreparation,
    create_all_gather_runtime,
    create_all_reduce_runtime,
    create_dma_all_reduce_runtime,
    create_roce_runtime,
)
from ignition.preparation.disposable import storage_ranges
from ignition.preparation.execution import GraphSignature
from ignition.preparation.frontend import PromptEncoder, TemplateEncoding, prepare_frontend
from ignition.preparation.host_memory import release_unused, virtual_device_memory
from ignition.preparation.model import PreparedModel
from ignition.preparation.rebuild_manifest import declare_weights
from ignition.preparation.scratch import ScratchArena
from ignition.preparation.tuning import RecordedPreparationSession, with_autotune_effort
from ignition.progress import phase

from hyperflare_recipes.components.multimax import apply_multimax
from hyperflare_recipes.components.paged_batch import BatchInputs, Metadata, PreparedBatch
from hyperflare_recipes.components.paged_generation import PreparedGeneration
from hyperflare_recipes.components.paged_maintenance import PreparedMaintenance
from hyperflare_recipes.components.paged_resume import PreparedResume
from hyperflare_recipes.components.parallel import TensorParallelShard
from hyperflare_recipes.components.prompt_logprobs import (
    PreparedScoredPrefill,
    PromptScoring,
    prefill_graph_count,
    prefill_modes,
)

from .attention import VHA_CHUNK_ROWS, VHA_RANK
from .config import VOCABULARY
from .contract import prepare as prepare_contract
from .decoder import PreparedDecoder, Scratch
from .dspark import make_forwards
from .pool import PAGE_TOKENS, ContinuationPool, continuation_layout, page_bytes
from .recipe import RECIPE
from .weights import ModelWeights


@dataclass(frozen=True)
class PreparationConfig(BasePreparationConfig):
    prompt_logprobs: bool = False
    devices: tuple[int, ...] = (0, 1)
    # Device ordinals are host-local when each rank runs on its own DGX Spark over RoCE.
    hosts: tuple[str, ...] | None = None
    roce_ring: bool = False
    roce_chain: bool = False
    context: int = 1048576
    dspark: bool = True
    draft_head_quantization: str | None = None
    markov_head_quantization: str | None = None
    speculative_graph_budget: int = 64
    packed_verification: bool | None = None
    draft_prefix_selection: str | None = None
    force_moe_decode_a16: bool = False
    expert_scale_compression: str | None = "csf"
    enable_flashindexer: bool = True
    stream_overlap: bool = False

    def __post_init__(self):
        if self.mtp_only and (
            not self.dspark
            or max(self.speculative_widths, default=1) <= 1
            or self.speculative_fixed_width == 1
        ):
            raise ValueError("--mtp-only requires enabled learned speculation with positive draft depth")
        if self.speculative_widths and max(self.speculative_widths) > 8:
            raise ValueError("ERNIE-5.5 DSpark draft model supports at most seven draft tokens")
        from ignition.declaration import Speculation
        from ignition.planning import host_shape_error

        Speculation(
            self.draft_tokens,
            draft_head_quantization=self.draft_head_quantization,
            markov_head_quantization=self.markov_head_quantization,
        )
        if self.draft_head_quantization not in (None, "nvfp4"):
            raise ValueError("ERNIE-5.5 draft head quantization must be omitted or nvfp4")
        if self.expert_scale_compression not in (None, "csf"):
            raise ValueError("ERNIE-5.5 expert scale compression must be None or 'csf'")
        drafting = self.dspark and self.draft_tokens > 0
        if self.packed_verification is None:
            object.__setattr__(self, "packed_verification", drafting)
        if self.draft_prefix_selection is None:
            object.__setattr__(self, "draft_prefix_selection", "gpu" if drafting else "cpu")
        if self.draft_prefix_selection not in ("cpu", "gpu"):
            raise ValueError("draft_prefix_selection must be 'cpu' or 'gpu'")
        placements = tuple(zip(self.hosts or ("",) * len(self.devices), self.devices))
        if (
            not isinstance(self.devices, tuple)
            or len(self.devices) not in (2, 4, 8)
            or len(set(placements)) != len(self.devices)
            or (
                self.hosts is not None
                and (not isinstance(self.hosts, tuple) or len(self.hosts) != len(self.devices))
            )
            or any(type(d) is not int or d < 0 for d in self.devices)
        ):
            raise ValueError("ERNIE-5.5 preparation requires 2, 4, or 8 distinct GPUs (dividing o_groups=8)")
        error = host_shape_error(
            self.roce_ring, self.roce_chain, None if self.hosts is None else len(self.hosts)
        )
        if error:
            raise ValueError(error)
        if not PAGE_TOKENS <= self.context <= 1048576 or self.context % PAGE_TOKENS:
            raise ValueError("ERNIE-5.5 context must be 256-aligned in [256, 1048576]")
        if not 64 <= self.prefill_chunk_size <= self.context or self.prefill_chunk_size % 64:
            raise ValueError("prefill chunk size must be 64-aligned in [64, context]")
        request_capacity(self.max_active)
        if (
            self.speculative_fixed_width is not None
            and self.speculative_fixed_width not in self.speculative_widths
        ):
            raise ValueError("fixed verification width must belong to the prepared widths")
        if self.snapshot_budget_bytes <= 0 or self.headroom_bytes < 0:
            raise ValueError("snapshot budget must be positive and headroom nonnegative")

    @property
    def request_capacity(self):
        return request_capacity(self.max_active)

    @property
    def prefill_capacities(self):
        return prefill_capacities(self.prefill_chunk_size)

    @property
    def batch_capacities(self):
        return tuple(reversed(batch_capacities(self.max_active)))

    @property
    def speculative_shapes(self):
        return (
            tuple(
                (batch, width)
                for batch, width in speculative_shapes(
                    self.max_active, self.speculative_widths, self.speculative_token_capacity
                )
                if width == 1
            )
            if self.dspark
            else ()
        )

    @property
    def packed_speculative_shapes(self):
        return (
            packed_speculative_shapes(
                self.max_active,
                max(self.speculative_widths),
                self.speculative_token_capacity,
                self.speculative_graph_budget,
            )
            if self.draft_tokens
            else ()
        )

    @property
    def speculative_round_shapes(self):
        from ignition.components.capacity import parallel_speculative_shapes

        return (
            parallel_speculative_shapes(
                self.max_active,
                self.speculative_widths,
                self.speculative_token_capacity,
                self.speculative_graph_budget,
                fixed_width=self.speculative_fixed_width,
            )
            if self.dspark
            else ()
        )

    @property
    def use_gpu_packed_speculation(self):
        return bool(
            self.dspark
            and self.draft_tokens > 0
            and self.packed_verification
            and self.draft_prefix_selection == "gpu"
            and self.speculative_fixed_width is None
        )

    @property
    def decode_tokens(self):
        if self.use_gpu_packed_speculation:
            return max(
                self.request_capacity,
                max((tokens for _, tokens in self.packed_speculative_shapes), default=0),
            )
        return max((packed for _, _, packed in self.speculative_round_shapes), default=self.request_capacity)

    @property
    def decode_samples(self):
        if self.use_gpu_packed_speculation:
            return max(
                self.request_capacity,
                max((batch * max(self.speculative_widths) for batch, _ in self.packed_speculative_shapes), default=0),
            )
        return max((batch * width for batch, width, _ in self.speculative_round_shapes), default=self.request_capacity)

    @property
    def draft_tokens(self):
        return max(self.speculative_widths) - 1 if self.dspark else 0

    @property
    def completion_token_capacity(self):
        if self.use_gpu_packed_speculation:
            return max(self.speculative_widths)
        return max((width for _, width, _ in self.speculative_round_shapes), default=1)


def _collective_runtimes(config: PreparationConfig, group, device, *, coordinator, core_storage):
    """Allocate hidden AR(4096), latent AR(2048), VHA FP32 gather, vocab FP32 gather, and Markov gather.

    When ranks span separate DGX Sparks over RoCE, one shared RoCEnante runtime
    serves every reduction, gather, and sampling broadcast in rank-serial order.
    """
    slots = config.request_capacity
    world = len(config.devices)
    tokens = max(config.prefill_chunk_size, config.decode_tokens, slots * config.draft_tokens)
    vocabulary_rows = max(config.decode_samples, slots * config.draft_tokens)
    local_vocab = VOCABULARY // world
    vha_rows = min(tokens, VHA_CHUNK_ROWS)
    # VHA FP32 partial per rank: [vha_rows, 16, 512] FP32 = vha_rows * 16 * 512 * 4 bytes.
    vha_gather_bytes = vha_rows * VHA_RANK * 512 * 4

    if coordinator.spans_hosts:
        from ignition.components.collectives import packed_broadcast_layout
        from hyperflare_recipes.components.draft_sampling import (
            PreparedDraftBlockBroadcast,
            PreparedDraftDistribution,
        )
        from hyperflare_recipes.components.speculation import SpeculativeAcceptance

        # FP32 vocabulary logits (4 bytes/elem) + VHA FP32 gather + Markov FP32 gather + broadcasts.
        gathers = [vocabulary_rows * local_vocab * 4, vha_gather_bytes]
        if config.dspark:
            batch = max(config.batch_capacities)
            gathers.append(slots * local_vocab * 4)
            gathers.extend(
                (packed_broadcast_layout(layout.values())[1] + 15) // 16 * 16
                for layout in (
                    PreparedDraftDistribution.broadcast_layout(batch),
                    SpeculativeAcceptance.broadcast_layout(batch, max(config.speculative_widths)),
                    PreparedDraftBlockBroadcast.broadcast_layout(batch, config.draft_tokens),
                )
            )
        runtime = create_roce_runtime(
            group,
            device,
            max_tokens=max(slots, config.decode_tokens, vocabulary_rows),
            pipe_tokens=tokens,
            hidden_size=4096,
            gather_bytes=max(gathers),
            ring=config.roce_ring,
            ring_open=config.roce_chain,
            core_storage=core_storage,
        )
        return runtime, runtime, runtime, runtime, (runtime if config.dspark else None), runtime

    ipc = core_storage.peer_storage
    collective = create_all_reduce_runtime(
        group,
        device,
        collective_tuning=config.collective_tuning,
        max_tokens=tokens,
        hidden_size=4096,
        ipc=ipc,
    )
    prefill = create_dma_all_reduce_runtime(
        group, device, max_tokens=config.prefill_chunk_size, hidden_size=4096, ipc=ipc
    )
    latent = create_all_reduce_runtime(
        group,
        device,
        collective_tuning=config.collective_tuning,
        max_tokens=tokens,
        hidden_size=2048,
        ipc=ipc,
    )
    # Vocabulary logits are FP32 (2x BF16 columns per vocabulary entry).
    vocabulary = create_all_gather_runtime(
        group, device, max_rows=vocabulary_rows, columns=2 * VOCABULARY, ipc=ipc
    )
    markov = (
        create_all_gather_runtime(group, device, max_rows=slots, columns=2 * VOCABULARY, ipc=ipc)
        if config.dspark
        else None
    )
    vha_gather = create_all_gather_runtime(
        group, device, max_rows=vha_rows, columns=world * 2 * VHA_RANK * 512, ipc=ipc
    )
    return collective, prefill, latent, vocabulary, markov, vha_gather


def _prepare_speculation_catalog(
    config: PreparationConfig,
    speculation,
    weights: ModelWeights,
    *,
    add,
    buffers,
    forward_inputs,
    host_allocate,
    device,
):
    if speculation is None:
        return None
    from hyperflare_recipes.components.parallel_draft import PreparedParallelDraftRound

    use_confidence = (
        config.use_gpu_packed_speculation
        and weights.draft is not None
        and weights.draft.confidence_head is not None
    )
    if use_confidence:
        from hyperflare_recipes.components.packed_speculation import PreparedPackedSpeculativeRound

        slots = config.request_capacity
        confidence = host_allocate("proposal_confidence", (2, slots, config.draft_tokens), torch.float32)
        buffers["proposal_confidence"] = confidence.device_view
        prefix_limits = torch.zeros(slots, dtype=torch.int32, device=device)
        buffers["speculative_prefix_limits"] = prefix_limits
        packed_inputs = tuple(field for field in forward_inputs if field != "query_lengths")
        packed_inputs += ("speculative_prefix_limits",)
        packed_batches = {batch for batch, _ in config.packed_speculative_shapes}
        proposals = {}
        for batch, depth in sorted(proposal_shapes(config.max_active, config.draft_tokens), reverse=True):
            if batch not in packed_batches:
                continue
            name = f"proposal{batch}x{depth + 1}"
            source = PreparedParallelDraftRound(speculation, batch, depth + 1, name, proposal_only=True)
            source.prepare()
            proposals[batch] = source
        for batch, _ in reversed(config.speculative_shapes):
            name = f"speculative{batch}x1"
            forward = PreparedParallelDraftRound(speculation, batch, 1, name)
            forward.prepare()
            add(name, forward, "speculative", batch, 1, forward_inputs, enabled=True)
        for batch, packed_tokens in sorted(
            config.packed_speculative_shapes, key=lambda shape: (shape[1], shape[0]), reverse=True
        ):
            logical_width = max(config.speculative_widths)
            target_name = f"verification{batch}x{logical_width}-tokens{packed_tokens}"
            target = PreparedParallelDraftRound(
                speculation,
                batch,
                logical_width,
                target_name,
                generate_proposals=False,
                target_token_capacity=packed_tokens,
            )
            target.prepare()
            name = f"packed-speculative{batch}x{logical_width}-drafts{config.draft_tokens}-tokens{packed_tokens}"
            forward = PreparedPackedSpeculativeRound(
                proposals[batch], target, prefix_limits, confidence.device_view, name=name
            )
            add(
                name,
                forward,
                "speculative",
                batch,
                logical_width,
                packed_inputs,
                enabled=True,
                packed_tokens=packed_tokens,
                draft_tokens=config.draft_tokens,
            )
        return confidence

    for batch, width, packed in sorted(
        config.speculative_round_shapes, key=lambda shape: (shape[2], shape[0]), reverse=True
    ):
        name = f"speculative{batch}x{width}-tokens{packed}"
        forward = PreparedParallelDraftRound(
            speculation, batch, width, name, target_token_capacity=packed
        )
        forward.prepare()
        add(name, forward, "speculative", batch, width, forward_inputs, enabled=True, packed_tokens=packed)
    return None


@torch.no_grad()
@with_autotune_effort
def prepare_rank(snapshot, config: PreparationConfig, *, rank, group, progress=None, core_storage=None):
    if core_storage is None:
        raise ValueError("ERNIE-5.5 preparation requires core storage")
    snapshot = Path(snapshot).resolve()
    shard = TensorParallelShard(rank, len(config.devices))
    device = torch.device("cuda", config.devices[rank])
    torch.cuda.set_device(device)
    with phase(progress, "Inspect checkpoint"):
        manifest = inspect_indexed_checkpoint(snapshot)
    metadata_path = manifest.metadata_path
    transient = contextlib.ExitStack()
    transient.enter_context(core_storage.transient())
    weights = ModelWeights.load(
        manifest,
        device,
        shard=shard,
        progress=progress,
        dspark=config.dspark,
        draft_head_quantization=config.draft_head_quantization,
        markov_head_quantization=config.markov_head_quantization,
        force_moe_decode_a16=config.force_moe_decode_a16,
        expert_scale_compression=config.expert_scale_compression,
        enable_flashindexer=config.enable_flashindexer,
        group=group,
    )
    if torch.cuda.get_device_properties(device).is_integrated:
        from ignition.preparation.host_memory import release_page_cache

        # Unified memory (DGX Spark) counts cached checkpoint pages against GPU headroom; drop them.
        release_page_cache(manifest.shard_paths)
    release_unused()
    sequence = torch.zeros(1, dtype=torch.int64, device=device)

    with phase(progress, "Allocate model state and collectives"):
        layout = continuation_layout(weights.config, dspark=config.dspark)
        snapshots = layout.snapshot_capacity(config.snapshot_budget_bytes)
        if snapshots < 1:
            raise ValueError("snapshot budget must hold a complete ERNIE-5.5 continuation")
        slots, blocks = config.request_capacity, config.context // PAGE_TOKENS
        execution_rows = slots * (max(config.speculative_widths) if config.dspark else 1)
        state_rows = snapshots + execution_rows
        initial_pages = (config.prefill_chunk_size + PAGE_TOKENS - 1) // PAGE_TOKENS + 2 * slots
        bytes_per_page = page_bytes(weights.config)
        total_memory = virtual_device_memory(device, group, tuning_replica=config.tuning_replica)
        pool = ContinuationPool.allocate(
            weights.config,
            state_rows + 1,
            initial_pages,
            device,
            storage=core_storage,
            maximum_pages=total_memory // bytes_per_page,
            maximum_rows=total_memory // layout.row_bytes,
            backed_slots=min(state_rows + 1, 1 + execution_rows + slots),
            dspark=config.dspark,
        )
        requests = DeviceRequestState.allocate(device, vocab_size=VOCABULARY, capacity=slots)
        inputs = BatchInputs.allocate(config.prefill_chunk_size, blocks, device, capacity=slots)
        admission = AdmissionBuffers.allocate(device, vocab_size=VOCABULARY, capacity=slots)
        commands = StateCommands.allocate(device, capacity=slots)
        frame = torch.zeros(1, dtype=torch.int32, device=device)
        completion = CompletionOutputs.allocate(
            (2, slots, VECTOR_TOKENS_FIELD + config.completion_token_capacity), device=device
        )
        coordinator = CollectivePreparation(group)
        (
            collective,
            prefill_collective,
            latent_collective,
            vocabulary_collective,
            markov_collective,
            vha_gather_collective,
        ) = _collective_runtimes(config, group, device, coordinator=coordinator, core_storage=core_storage)

        identity = hashlib.sha256(
            json.dumps(
                {
                    "config": json.loads((metadata_path / "config.json").read_text()),
                    "headers": [s.header_sha256 for s in manifest.shards],
                    "world_size": shard.world_size,
                    "contract": "ernie55.graph-inputs.v1",
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()
        session = RecordedPreparationSession(
            compile_workers=config.compile_workers,
            collective_tuning=config.collective_tuning,
            collective_ladder=config.collective_ladder,
            autotune_effort=config.autotune_effort,
            tuning_replica=config.tuning_replica,
            native_dense_activations=config.native_dense_activations,
            lock_gpu_clocks=config.lock_gpu_clocks,
            device=device,
            checkpoint=identity,
            workload_schema="ernie55-flash",
            numerical_policy="default",
            coordinator=coordinator,
            headroom_bytes=config.headroom_bytes,
            benchmark_storage=core_storage,
        )
        session.__enter__()
        session.register_benchmark_resource("continuations", pool, restore=pool.clear_for_preparation)
        examples = {}
        dspark_forwards = make_forwards(
            latent_runtime=latent_collective,
            vha_gather_runtime=vha_gather_collective,
            examples=examples,
        )
        arena = ScratchArena(device, overlap=config.stream_overlap)
        session.register_scratch(arena)
        prefill_storage = Scratch.allocate(
            config.prefill_chunk_size, 1, shard.world_size, device, blocks=blocks
        )
        decoder_storage = Scratch.allocate(
            config.decode_tokens, config.decode_samples, shard.world_size, device, blocks=blocks
        )
        buffers = dict(
            inputs.buffers(),
            **commands.buffers(),
            **{name: getattr(admission, name) for name in admission.__dataclass_fields__},
            state_pool=pool.backing,
            memory_capacity=pool.memory_capacity,
            sampling_grammar_mask=requests.grammar_mask,
            frame_index=frame,
            submission_sequence=sequence,
            **completion.buffers(),
        )
        forwards, signatures, metadata, scratch = {}, {}, {}, {}
        speculation = None
        if config.dspark:
            from hyperflare_recipes.components.dspark import SpeculativePreparation

            speculation = SpeculativePreparation(
                weights,
                None,
                pool,
                requests,
                inputs,
                decoder_storage,
                draft_tokens=config.draft_tokens,
                completion=completion,
                frame=frame,
                sequence=sequence,
                session=session,
                workspace=arena,
                runtime=collective,
                vocabulary_runtime=vocabulary_collective,
                markov_runtime=markov_collective,
                coordinator=coordinator,
                forwards=dspark_forwards,
                page_tokens=PAGE_TOKENS,
            )

    scoring = PromptScoring.allocate(
        config,
        device,
        buffers,
        frame=frame,
        group=group,
        ipc=core_storage.peer_storage,
        runtime=collective,
        coordinator=coordinator,
    )
    del manifest
    modes = serving_modes(config.dspark, config.mtp_only)
    spec_count = (
        len(config.speculative_shapes) + len(config.packed_speculative_shapes)
        if config.use_gpu_packed_speculation
        and weights.draft is not None
        and weights.draft.confidence_head is not None
        else len(config.speculative_round_shapes)
    )
    count = (
        len(modes) * (prefill_graph_count(config.prefill_capacities, scoring) + 4 * len(config.batch_capacities))
        + spec_count
        + (len(config.batch_capacities) if False in modes else 0)
    )
    with phase(progress, "Construct graph catalog", total=count) as stage:

        def add(name, forward, kind, rows, tokens, fields, *, enabled=None, packed_tokens=0, draft_tokens=None):
            if kind == "scored_prefill":
                forward = PreparedScoredPrefill(forward, requests, inputs)
                fields = (*fields, "prompt_target_tokens")
            forwards[name] = forward
            signatures[name] = GraphSignature(
                kind,
                rows,
                tokens,
                tuple(fields),
                speculation_enabled=enabled,
                packed_token_capacity=packed_tokens,
                draft_tokens=draft_tokens,
                confidence_width=draft_tokens if "speculative_prefix_limits" in fields and draft_tokens else 0,
            )
            stage.update(advance=1)

        forward_inputs = (*inputs.buffers(), "sampling_grammar_mask")
        for enabled in modes:
            suffix = "-no-mtp" if config.dspark and not enabled else ""
            mode_weights = weights if enabled else replace(weights, draft=None)
            for kind, mode, capacities in prefill_modes(
                config.prefill_capacities, () if enabled else config.batch_capacities, scoring
            ):
                for capacity in capacities:
                    rows = 1 if mode == "prefill" else capacity
                    name = f"{kind}{capacity}" + (suffix if mode == "prefill" else "")
                    metadata[name] = Metadata.allocate(capacity, rows, blocks, device)
                    storage = prefill_storage if mode == "prefill" else decoder_storage
                    scratch[name] = storage.rows(capacity, rows)
                    decoder = PreparedDecoder(
                        mode_weights,
                        None,
                        scratch[name],
                        metadata[name],
                        pool,
                        mode=mode,
                        scoring=scoring if kind == "scored_prefill" else None,
                        runtime=prefill_collective if mode == "prefill" and capacity >= 1024 else collective,
                        latent_runtime=latent_collective,
                        vha_gather_runtime=vha_gather_collective,
                        vocabulary_runtime=vocabulary_collective,
                        coordinator=coordinator,
                        session=session,
                        name=name,
                        workspace=arena,
                        examples=examples,
                    )
                    batch = PreparedBatch(
                        inputs,
                        requests,
                        metadata[name],
                        mode=mode,
                        shard=shard,
                        pool=pool,
                        page_tokens=PAGE_TOKENS,
                        input_vocabulary=VOCABULARY,
                    )
                    forward = PreparedGeneration(
                        decoder,
                        batch,
                        completion,
                        frame,
                        sequence,
                        runtime=collective,
                        coordinator=coordinator,
                        session=session,
                        name=name + ".generation",
                        workspace=arena,
                        context_factory=dspark_forwards.context,
                    )
                    add(
                        name,
                        forward,
                        mode,
                        rows,
                        capacity if mode == "prefill" else 1,
                        forward_inputs,
                        enabled=enabled,
                    )

        confidence = _prepare_speculation_catalog(
            config,
            speculation,
            weights,
            add=add,
            buffers=buffers,
            forward_inputs=forward_inputs,
            host_allocate=core_storage.host_allocate,
            device=device,
        )

        row_inputs = ("request_state_rows", "row_slot_ids", "row_generations", "live_rows")
        multimax_post = partial(
            apply_multimax, ranges=weights.multimax_ranges, ts=weights.multimax_ts
        )
        for enabled in modes:
            suffix = "-no-mtp" if config.dspark and not enabled else ""
            mode_weights = weights if enabled else replace(weights, draft=None)
            for capacity in config.batch_capacities:
                name = f"resume{capacity}" + suffix
                forward = PreparedResume(
                    mode_weights,
                    requests,
                    pool,
                    inputs,
                    completion,
                    frame,
                    sequence,
                    rows=capacity,
                    runtime=collective,
                    vocabulary_runtime=vocabulary_collective,
                    coordinator=coordinator,
                    session=session,
                    name=name,
                    logits_dtype=torch.float32,
                    workspace=arena,
                    postprocess_logits=multimax_post,
                )
                add(
                    name,
                    forward,
                    "resume",
                    capacity,
                    1,
                    (*row_inputs, "sampling_grammar_mask"),
                    enabled=enabled,
                )
                for mode in ("init", "restore", "snapshot"):
                    name = f"{mode}{capacity}" + suffix
                    forward = PreparedMaintenance(
                        mode,
                        requests,
                        pool,
                        commands,
                        inputs,
                        completion,
                        frame,
                        sequence,
                        runtime=collective,
                        coordinator=coordinator,
                        session=session,
                        name=name,
                        capacity=capacity,
                        admission_buffers=admission,
                        draft_state=enabled,
                        page_tokens=PAGE_TOKENS,
                    )
                    fields = (
                        row_inputs
                        if mode == "snapshot"
                        else ("request_state_rows", *admission.__dataclass_fields__)
                    )
                    add(
                        name,
                        forward,
                        mode,
                        capacity,
                        1,
                        (*fields, "page_tables", *commands.buffers()),
                        enabled=enabled,
                    )

    if progress is not None:
        progress(
            f"before tuning: {torch.cuda.memory_allocated(device) >> 20} MiB allocated, "
            f"{torch.cuda.mem_get_info(device)[0] >> 20} MiB free"
        )
    with phase(progress, "Prepare and tune kernels") as stage, transient:
        tuning = session.tune(show_progress=progress is not None and rank == 0, console=stage.console)
    if speculation is not None:
        speculation.release_examples()
    examples.clear()
    metadata.clear()
    scratch.clear()
    del prefill_storage, decoder_storage, storage, decoder, batch
    arena.release_preparation()

    stream, graph_pool = torch.cuda.Stream(device=device), core_storage.graph_pool.id
    pages = initial_pages

    def materialize_cache():
        pool.clear_for_preparation()

    graphs, compiled = capture_forwards(
        forwards,
        signatures,
        session=session,
        device=device,
        stream=stream,
        pool=graph_pool,
        collective=collective,
        coordinator=coordinator,
        compile_decoder=config.compile_decoder,
        progress=progress,
        cache_root=config.kernel_cache,
        retain_graphs=config.retain_graphs,
        materialize=materialize_cache,
        warmup_storage=core_storage,
        compile_kinds=("prefill", "scored_prefill", "decode", "resume", "speculative"),
    )
    session.freeze()
    pool.clear_for_preparation()
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    release_unused()
    if group is not None:
        torch.distributed.barrier(group=group)
    free, total = torch.cuda.mem_get_info(device)

    assets_dir = Path(__file__).parent / "assets"
    assets = {
        "tokenizer": {"path": str(metadata_path / "tokenizer.json")},
        "chat_template": {"path": str(assets_dir / "chat_template.jinja")},
    }
    buffers["request_pending_token"] = requests.pending_token
    server_config = {
        "identity": identity,
        "model_id": snapshot.name,
        "max_sequence_tokens": config.context,
        "prefill_chunk_size": config.prefill_chunk_size,
        "pages": pages,
        "state_rows": state_rows,
        "snapshots": snapshots,
        "page_tokens": PAGE_TOKENS,
        "frontend": prepare_frontend(
            metadata_path,
            eos_token_ids=[2, 8],
            prompt_encoder=PromptEncoder.PROMPT_JINJA,
            template_encoding=TemplateEncoding.TEMPLATE_TOKEN_SEGMENTS,
        ),
        "sampling_defaults": RECIPE.sampling_defaults.as_dict(),
        "assets": assets,
        "model_contract": prepare_contract(
            signatures,
            buffers,
            pool,
            page_tokens=PAGE_TOKENS,
            draft_lag=0 if config.dspark else None,
            rows=None,
        ),
    }
    memory = {
        "rank": rank,
        "world_size": shard.world_size,
        "device": config.devices[rank],
        "weights_bytes": weights.allocated_bytes,
        "draft_head_quantization": config.draft_head_quantization,
        "markov_head_quantization": config.markov_head_quantization,
        "force_moe_decode_a16": config.force_moe_decode_a16,
        "expert_scale_compression": config.expert_scale_compression,
        "mapped_host_bytes": 0,
        "offloaded_bytes": 0,
        "continuation_bytes": pool.backing.nbytes,
        "preparation_state_rows": int(pool.memory_capacity[0].item()),
        "preparation_state_bytes": int(pool.memory_capacity[0].item()) * layout.row_bytes,
        "state_rows": state_rows,
        "pages": pages,
        "page_bytes": bytes_per_page,
        "page_tokens": PAGE_TOKENS,
        "kv_bytes": pages * bytes_per_page,
        "completion_output_bytes": completion.nbytes,
        "operator_scratch_bytes": arena.nbytes,
        "graphs": len(graphs),
        "prefill_capacities": config.prefill_capacities,
        "free_device_bytes": free,
        "total_device_bytes": total,
        "headroom_bytes": config.headroom_bytes,
        "allocated_bytes": torch.cuda.memory_allocated(device),
        "reserved_bytes": torch.cuda.memory_reserved(device),
        "tuning": tuning,
        "compiled_decoder": config.compile_decoder,
    }
    if scoring is not None:
        memory.update(scoring.memory_report())
    if config.dspark:
        widths = (
            tuple(config.speculative_widths)
            if confidence is not None
            else tuple(sorted({width for _, width, _ in config.speculative_round_shapes}))
        )
        server_config.update(
            speculative_widths=widths,
            speculative_token_capacity=speculative_token_capacity(
                config.max_active, max(config.speculative_widths), config.speculative_token_capacity
            ),
            speculative_fixed_width=config.speculative_fixed_width,
        )
        memory.update(
            speculative_round_shapes=(
                config.packed_speculative_shapes if confidence is not None else config.speculative_round_shapes
            ),
            speculative_graph_budget=config.speculative_graph_budget,
        )
    disposable = []
    for forward in forwards.values():
        if isinstance(forward, PreparedScoredPrefill):
            forward = forward.forward
        if hasattr(forward, "completion") and hasattr(forward.completion, "scratch_tensors"):
            disposable.append(forward.completion.scratch_tensors)
    model = PreparedModel(
        rebuild_weights=declare_weights(
            weights,
            components=("embedding", "norm", "head", "multimax_ranges", "multimax_ts", "head_mix", "draft"),
        ),
        device=config.devices[rank],
        graphs=graphs,
        buffers=buffers,
        signatures=signatures,
        max_active=config.max_active,
        session=session,
        server_config=server_config,
        memory_report=memory,
        export_zero_ranges=(
            *(() if scoring is None else scoring.zero_ranges()),
            *storage_ranges(disposable),
            *completion.zero_ranges(),
        ),
        resources=(
            scoring,
            weights,
            pool,
            requests,
            inputs,
            admission,
            commands,
            arena,
            forwards,
            scratch,
            metadata,
            compiled,
            stream,
            graph_pool,
            session,
            collective,
            prefill_collective,
            latent_collective,
            vocabulary_collective,
            markov_collective,
            vha_gather_collective,
            coordinator,
            speculation,
            confidence,
        ),
        close_resources=tuple(
            dict.fromkeys(
                c
                for c in (
                    scoring,
                    markov_collective,
                    vha_gather_collective,
                    latent_collective,
                    prefill_collective,
                    vocabulary_collective,
                    collective,
                )
                if c is not None
            )
        ),
    )
    if confidence is not None:
        model.proposal_confidence = confidence.host_view
    return model
