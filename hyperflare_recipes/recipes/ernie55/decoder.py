# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""ERNIE-5.5 decoder: mHC residual streams, hybrid HCA/DSA attention, Latent MoE, and multimax LM head."""

from dataclasses import dataclass, fields

import torch

from ignition.components.collectives import PreparedAllReduce
from hyperflare_recipes.components.dspark import PreparedFeedbackCapture
from hyperflare_recipes.components.embedding import PreparedEmbedding
from hyperflare_recipes.components.mhc import (
    MHCScratch,
    MHCWeights,
    PreparedMHC,
    PreparedMHCHead,
    PreparedMHCPost,
)
from hyperflare_recipes.components.mlp import MLPScratch
from hyperflare_recipes.components.multimax import apply_multimax
from hyperflare_recipes.components.rms_norm import PreparedRMSNorm
from hyperflare_recipes.components.rotary import CompressedRotary as Rotary
from hyperflare_recipes.components.situ import PreparedSiTUMLP
from hyperflare_recipes.components.vocabulary import PreparedVocabularyProjection

from .attention import (
    AttentionIndices,
    AttentionScratch,
    PreparedDSAAttention,
    PreparedHCAAttention,
)
from .config import VOCABULARY
from .moe import LatentMoEScratch, PreparedLatentMoE


@dataclass(frozen=True)
class Scratch:
    embedding: torch.Tensor
    residual: torch.Tensor
    attention_hc: MHCScratch
    ffn_hc: MHCScratch
    attention: AttentionScratch
    attention_output: torch.Tensor
    dense_mlp: MLPScratch
    moe: LatentMoEScratch
    head_residual: torch.Tensor
    collapsed: torch.Tensor
    hidden: torch.Tensor
    local_logits: torch.Tensor
    logits: torch.Tensor

    def rows(self, tokens: int, samples: int):
        def select(value, count):
            if isinstance(value, torch.Tensor):
                return value[:count]
            return type(value)(*(select(getattr(value, f.name), count) for f in fields(value)))

        sampled = ("head_residual", "collapsed", "hidden", "local_logits", "logits")
        return type(self)(
            *(
                select(getattr(self, f.name), samples if f.name in sampled else tokens)
                for f in fields(self)
            )
        )

    @classmethod
    def allocate(
        cls,
        tokens: int,
        samples: int,
        world: int,
        device,
        *,
        blocks: int = 4096,
        experts: int = 512,
        top_k: int = 10,
        logits_dtype: torch.dtype = torch.float32,
        include_dense_mlp: bool = True,
    ):
        def zeros(*shape, dtype=torch.bfloat16):
            return torch.zeros(shape, dtype=dtype, device=device)

        dense_inter = 14336 // world if include_dense_mlp else 2048 // world
        return cls(
            embedding=zeros(tokens, 4096),
            residual=zeros(tokens, 4, 4096),
            attention_hc=MHCScratch.allocate(tokens, 4096, device),
            ffn_hc=MHCScratch.allocate(tokens, 4096, device),
            attention=AttentionScratch.allocate(tokens, world, device, blocks=blocks),
            attention_output=zeros(tokens, 4096),
            dense_mlp=MLPScratch.allocate(tokens, 4096, dense_inter, device),
            moe=LatentMoEScratch.allocate(tokens, world, device, experts=experts, top_k=top_k),
            head_residual=zeros(samples, 4, 4096),
            collapsed=zeros(samples, 4096),
            hidden=zeros(samples, 4096),
            local_logits=zeros(samples, VOCABULARY // world, dtype=logits_dtype),
            logits=zeros(samples, VOCABULARY, dtype=logits_dtype),
        )


class PreparedLayer:
    def __init__(
        self,
        index: int,
        weights,
        scratch: Scratch,
        metadata,
        pool,
        rotary,
        *,
        shard,
        mode: str,
        runtime,
        latent_runtime,
        vha_gather_runtime,
        coordinator,
        session,
        name: str,
        workspace,
        carry: bool,
        defer_post: bool,
        indices: AttentionIndices,
        examples=None,
    ):
        s, cfg = scratch, pool.config
        text = cfg.text
        self.carry = carry
        options = {
            "rms_eps": text["rms_norm_eps"],
            "norm_eps": text["rms_norm_eps"],
            "hc_eps": text["hc_eps"],
            "sinkhorn_iters": text["hc_sinkhorn_iters"],
        }

        def mhc_weights(kind: str, norm: str):
            r = weights.residual
            return MHCWeights(
                r[f"hc_{kind}_fn"], r[f"hc_{kind}_scale"], r[f"hc_{kind}_base"], r[norm + ".weight"]
            )

        incoming = (
            (s.moe.mlp_output, s.ffn_hc.residual, s.ffn_hc.post, s.ffn_hc.comb)
            if carry
            else (s.residual,)
        )
        self.attention_hc = PreparedMHC(
            incoming,
            mhc_weights("attn", "attn_norm"),
            s.attention_hc,
            session=session,
            name=name + ".attn_hc",
            workspace=workspace,
            max_rows_per_call=2048 if mode == "prefill" and carry else None,
            **options,
        ).functional()

        if weights.is_dsa:
            self.attention = PreparedDSAAttention(
                index,
                weights.attention,
                s.attention_hc.normalized,
                s.attention,
                metadata,
                pool,
                rotary,
                mode=mode,
                world=shard.world_size,
                vha_gather_runtime=vha_gather_runtime,
                coordinator=coordinator,
                session=session,
                name=name + ".attention",
                workspace=workspace,
                indices=indices,
                examples=examples,
            )
        else:
            self.attention = PreparedHCAAttention(
                index,
                weights.attention,
                s.attention_hc.normalized,
                s.attention,
                metadata,
                pool,
                rotary,
                mode=mode,
                world=shard.world_size,
                vha_gather_runtime=vha_gather_runtime,
                coordinator=coordinator,
                session=session,
                name=name + ".attention",
                workspace=workspace,
                indices=indices,
                examples=examples,
            )

        self.attention_reduce = PreparedAllReduce(
            s.attention.output,
            s.attention_output,
            runtime=runtime,
            coordinator=coordinator,
            session=session,
            name=name + ".attention_reduce",
        ).functional(session=session)

        self.ffn_hc = PreparedMHC(
            (s.attention_output, s.attention_hc.residual, s.attention_hc.post, s.attention_hc.comb),
            mhc_weights("ffn", "ffn_norm"),
            s.ffn_hc,
            session=session,
            name=name + ".ffn_hc",
            workspace=workspace,
            max_rows_per_call=2048 if mode == "prefill" else None,
            **options,
        ).functional()

        self.dense_mlp = None
        self.dense_reduce = None
        self.moe = None
        if weights.mlp is not None:
            self.dense_mlp = PreparedSiTUMLP(
                s.ffn_hc.normalized,
                weights.mlp,
                s.dense_mlp.gate_up,
                s.dense_mlp.intermediate,
                s.dense_mlp.output,
                session=session,
                name=name + ".mlp",
                workspace=workspace,
                beta=cfg.situ_beta,
                linear_beta=cfg.situ_linear_beta,
                activation_mode="auto",
                examples=examples,
                example_role="mlp.dense",
            ).functional(session=session)
            self.dense_reduce = PreparedAllReduce(
                s.dense_mlp.output,
                s.moe.mlp_output,
                runtime=runtime,
                coordinator=coordinator,
                session=session,
                name=name + ".mlp_reduce",
            ).functional(session=session)
        else:
            self.moe = PreparedLatentMoE(
                weights.experts,
                s.ffn_hc.normalized,
                s.moe,
                shard=shard,
                eps=text["rms_norm_eps"],
                beta=cfg.situ_beta,
                linear_beta=cfg.situ_linear_beta,
                top_k=text["num_experts_per_tok"],
                latent_runtime=latent_runtime,
                hidden_runtime=runtime,
                coordinator=coordinator,
                session=session,
                name=name + ".moe",
                workspace=workspace,
                examples=examples,
            )

        self.post = (
            None
            if defer_post
            else PreparedMHCPost(
                s.moe.mlp_output,
                s.ffn_hc.residual,
                s.ffn_hc.post,
                s.ffn_hc.comb,
                s.residual,
                session=session,
                name=name + ".post",
            ).functional()
        )

    def __call__(self, incoming, metadata, indices):
        residual, post, comb, source = self.attention_hc(*incoming)
        attended, records = self.attention(source, metadata, indices)
        attended = self.attention_reduce(attended)
        residual, post, comb, normalized = self.ffn_hc(attended, residual, post, comb)
        if self.dense_mlp is not None:
            output = self.dense_reduce(self.dense_mlp(normalized))
        else:
            output = self.moe(normalized, metadata)
        if self.post is None:
            return (output, residual, post, comb), records
        result = self.post(output, residual, post, comb, torch.empty_like(residual))
        return result, records


class PreparedDecoder:
    def __init__(
        self,
        weights,
        tables,
        scratch: Scratch,
        metadata,
        pool,
        *,
        mode: str,
        runtime,
        latent_runtime,
        vha_gather_runtime,
        vocabulary_runtime,
        coordinator,
        session,
        name: str,
        workspace,
        scoring=None,
        examples=None,
    ):
        del tables
        self.weights, self.scratch, self.pool = weights, scratch, pool
        self.mode = mode
        self.scoring = scoring
        self.aux_layers = () if weights.draft is None else weights.config.draft.target_layers
        self.feedback = (
            PreparedFeedbackCapture(
                metadata, len(self.aux_layers) * 4096, compact_prefill=mode == "prefill"
            )
            if self.aux_layers
            else None
        )
        feedback_rows = (
            min(128, metadata.token_ids.numel())
            if mode == "prefill"
            else metadata.token_ids.numel()
        )
        self.auxiliary = (
            None
            if self.feedback is None
            else torch.empty(
                (feedback_rows, len(self.aux_layers) * 4096),
                dtype=torch.bfloat16,
                device=scratch.embedding.device,
            )
        )
        safe_tokens = torch.zeros_like(metadata.token_ids)
        self.embedding = PreparedEmbedding(
            weights.embedding,
            safe_tokens,
            scratch.embedding,
            num_rows=metadata.live_counts[:1],
            session=session,
            name=name + ".embedding",
        ).functional(session=session, dynamic_inputs=True)
        self.rotary = Rotary(weights.config, scratch.embedding.device)
        self.indices = AttentionIndices(
            metadata, mode=mode, tokens=metadata.token_ids.numel(), has_dsa=True, examples=examples
        )
        self.layers, carry = [], False
        for index, layer in enumerate(weights.layers):
            defer = index + 1 < len(weights.layers) and index not in self.aux_layers
            self.layers.append(
                PreparedLayer(
                    index,
                    layer,
                    scratch,
                    metadata,
                    pool,
                    self.rotary,
                    shard=weights.shard,
                    mode=mode,
                    runtime=runtime,
                    latent_runtime=latent_runtime,
                    vha_gather_runtime=vha_gather_runtime,
                    coordinator=coordinator,
                    session=session,
                    name=f"{name}.layer{index}",
                    workspace=workspace,
                    carry=carry,
                    defer_post=defer,
                    indices=self.indices,
                    examples=examples,
                )
            )
            carry = defer
        self.collapse = PreparedMHCHead(
            scratch.head_residual,
            weights.head_mix,
            scratch.collapsed,
            rms_eps=weights.config.text["rms_norm_eps"],
            hc_eps=weights.config.text["hc_eps"],
            session=session,
            name=name + ".collapse",
        ).functional()
        self.norm = PreparedRMSNorm(
            scratch.collapsed,
            weights.norm,
            scratch.hidden,
            eps=weights.config.text["rms_norm_eps"],
            session=session,
            name=name + ".norm",
        ).functional()
        self.scoring = (
            None
            if scoring is None
            else scoring.prepare(
                scratch.embedding,
                weights.head,
                vocabulary=VOCABULARY,
                session=session,
                coordinator=coordinator,
                name=name + ".prompt",
            )
        )
        self.score_collapse = (
            None
            if self.scoring is None
            else PreparedMHCHead(
                scratch.residual,
                weights.head_mix,
                scratch.embedding,
                rms_eps=weights.config.text["rms_norm_eps"],
                hc_eps=weights.config.text["hc_eps"],
                session=session,
                name=name + ".prompt_collapse",
            ).functional()
        )
        self.score_norm = (
            None
            if self.scoring is None
            else PreparedRMSNorm(
                scratch.embedding,
                weights.norm,
                scratch.embedding,
                eps=weights.config.text["rms_norm_eps"],
                session=session,
                name=name + ".prompt_norm",
            ).functional()
        )
        self.head = PreparedVocabularyProjection(
            scratch.hidden,
            weights.head,
            scratch.local_logits,
            scratch.logits,
            runtime=vocabulary_runtime,
            coordinator=coordinator,
            session=session,
            name=name + ".head",
            workspace=workspace,
        ).functional(session=session)
        self.indices.functional()

    def release_outputs(self):
        self.scratch = self.auxiliary = None

    def __call__(self, metadata):
        embedding = self.embedding(
            token_ids=metadata.token_ids.clamp(0, VOCABULARY - 1),
            num_rows=metadata.live_counts[:1],
        )
        residual = embedding[:, None, :].expand(-1, 4, -1).clone()
        indices = self.indices(metadata)
        incoming = (residual,)
        records = []
        feedback_state = None if self.feedback is None else self.feedback.begin(metadata)
        aux_col = 0
        for index, layer in enumerate(self.layers):
            result, layer_records = layer(incoming, metadata, indices)
            records.append(layer_records)
            if feedback_state is not None and index in self.aux_layers:
                feedback_state = self.feedback(feedback_state, result, aux_col)
                aux_col += 4096
            incoming = result if isinstance(result, tuple) else (result,)
        residual = result
        hidden = self.norm(self.collapse(residual[metadata.sample_indices]))
        if self.scoring is not None:
            self.scoring(self.score_norm(self.score_collapse(residual)))
        logits = apply_multimax(
            self.head(hidden), self.weights.multimax_ranges, self.weights.multimax_ts
        )
        auxiliary = None if feedback_state is None else feedback_state[0]
        return hidden, logits, auxiliary, tuple(records)

    def commit(self, counts, records, metadata):
        for layer, layer_records in zip(self.layers, records, strict=True):
            if layer_records is not None:
                layer.attention.commit(counts, layer_records, metadata)
