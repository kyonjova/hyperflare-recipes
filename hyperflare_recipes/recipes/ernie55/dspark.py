# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Bundled 3-layer DSpark draft model with post-Markov multimax and confidence head."""

from functools import partial

import torch

from hyperflare_recipes.components.dspark import DSparkForwards, PreparedContext, PreparedMarkov
from hyperflare_recipes.components.embedding import PreparedEmbedding
from hyperflare_recipes.components.mhc import PreparedMHCHead
from hyperflare_recipes.components.multimax import MultimaxSelector
from hyperflare_recipes.components.rms_norm import PreparedRMSNorm
from hyperflare_recipes.components.rotary import CompressedRotary
from hyperflare_recipes.components.vocabulary import PreparedVocabularyProjection

from .attention import AttentionIndices
from .decoder import PreparedDecoder, PreparedLayer, Scratch


class DraftScratch(Scratch):
    """Scratch allocation for the 3-layer DSpark draft (BF16 proposal logits, no dense MLP)."""

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
    ):
        return super().allocate(
            tokens,
            samples,
            world,
            device,
            blocks=blocks,
            experts=experts,
            top_k=top_k,
            logits_dtype=torch.bfloat16,
            include_dense_mlp=False,
        )


class PreparedDraft:
    def __init__(
        self,
        weights,
        scratch: Scratch,
        metadata,
        pool,
        *,
        runtime,
        latent_runtime,
        vha_gather_runtime,
        vocabulary_runtime,
        coordinator,
        session,
        name: str,
        workspace,
        examples=None,
    ):
        self.weights, self.scratch = weights, scratch
        # Attach MultimaxSelector so PreparedParallelDraftRound applies multimax
        # after Markov bias (preserving -inf masked entries) without fused shortcuts.
        self.selector = MultimaxSelector(weights.multimax_ranges, weights.multimax_ts)
        self.embedding = PreparedEmbedding(
            weights.embedding,
            metadata.token_ids,
            scratch.embedding,
            num_rows=metadata.live_counts[:1],
            session=session,
            name=name + ".embedding",
        ).functional(session=session, dynamic_inputs=True)
        rotary = CompressedRotary(weights.config, scratch.embedding.device)
        self.indices = AttentionIndices(
            metadata, mode="draft", tokens=metadata.token_ids.numel(), has_dsa=False
        )
        self.layers = tuple(
            PreparedLayer(
                i,
                layer,
                scratch,
                metadata,
                pool,
                rotary,
                shard=weights.shard,
                mode="draft",
                runtime=runtime,
                latent_runtime=latent_runtime,
                vha_gather_runtime=vha_gather_runtime,
                coordinator=coordinator,
                session=session,
                name=f"{name}.layer{i}",
                workspace=workspace,
                carry=i > 0,
                defer_post=i + 1 < len(weights.draft.layers),
                indices=self.indices,
                examples=examples,
            )
            for i, layer in enumerate(weights.draft.layers)
        )
        self.head_hidden = scratch.collapsed
        self.head_hidden_shape = tuple(scratch.collapsed.shape)
        self.logit_shape = tuple(scratch.logits.shape)
        self.local_logit_shape = tuple(scratch.local_logits.shape)
        self.logit_dtype = scratch.logits.dtype
        self.local_logit_dtype = scratch.local_logits.dtype
        self.collapse = PreparedMHCHead(
            scratch.head_residual,
            weights.draft.head_mix,
            self.head_hidden,
            rms_eps=weights.config.text["rms_norm_eps"],
            hc_eps=weights.config.text["hc_eps"],
            session=session,
            name=name + ".collapse",
        ).functional()
        self.norm = PreparedRMSNorm(
            self.head_hidden,
            weights.draft.norm,
            scratch.hidden,
            eps=weights.config.text["rms_norm_eps"],
            session=session,
            name=name + ".norm",
        ).functional()
        self.head = PreparedVocabularyProjection(
            scratch.hidden,
            weights.head if weights.draft.draft_head is None else weights.draft.draft_head,
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
        self.scratch = self.head_hidden = None

    def __call__(self, metadata):
        embedding = self.embedding(
            token_ids=metadata.token_ids, num_rows=metadata.live_counts[:1]
        )
        residual = embedding[:, None, :].expand(-1, 4, -1).clone()
        indices = self.indices(metadata)
        incoming = (residual,)
        for layer in self.layers:
            result, _ = layer(incoming, metadata, indices)
            incoming = result if isinstance(result, tuple) else (result,)
        head_hidden = self.collapse(result[metadata.sample_indices])
        logits = self.head(self.norm(head_hidden))
        batch = metadata.starts.numel()
        return logits, None, {"head_hidden": head_hidden.view(batch, -1, head_hidden.shape[-1])}


def make_forwards(*, latent_runtime, vha_gather_runtime, examples=None):
    return DSparkForwards(
        decoder=partial(
            PreparedDecoder,
            latent_runtime=latent_runtime,
            vha_gather_runtime=vha_gather_runtime,
            examples=examples,
        ),
        scratch=DraftScratch,
        context=partial(PreparedContext, cache_format="deepseek_v4", compact_prefill=True),
        draft=partial(
            PreparedDraft,
            latent_runtime=latent_runtime,
            vha_gather_runtime=vha_gather_runtime,
            examples=examples,
        ),
        markov=PreparedMarkov,
    )
