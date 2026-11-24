# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Hybrid C128 HCA and full-token DSA paged cache and continuation state layout."""

from dataclasses import dataclass

import torch

from hyperflare_recipes.components.paged_state import PagedStateStorage
from ignition.components.state import StateLayout

from .config import DSA_LAYER_RATIO, HCA_LAYER_RATIO

PAGE_TOKENS = 256
# GLM_NSA FP8 latent record: 512 B E4M3 NoPE + 16 B 4xFP32 scales + 128 B BF16 RoPE.
DSA_RECORD_BYTES = 656
INDEX_RECORD_BYTES = 132
INDEX_SUBPAGE_TOKENS = 64


def hca_latent_page_bytes(records: int) -> int:
    """DeepSeek-V4 C128/SWA cache layout: 576 B payload + 8 B UE8M0 scale footer, 576-aligned."""
    return ((records * 584 + 575) // 576) * 576


def page_specs(config):
    """Per-page byte regions across all 37 HCA (C128) and 6 DSA layers."""
    for layer, ratio in enumerate(config.text["compress_ratios"][: config.layers]):
        if ratio == HCA_LAYER_RATIO:
            yield f"layer{layer}.latent_page", (hca_latent_page_bytes(PAGE_TOKENS // ratio),)
        elif ratio == DSA_LAYER_RATIO:
            yield f"layer{layer}.latent_page", (PAGE_TOKENS * DSA_RECORD_BYTES,)
            yield f"layer{layer}.index_page", (PAGE_TOKENS * INDEX_RECORD_BYTES,)
        else:
            raise ValueError(f"unsupported ERNIE-5.5 compress_ratio {ratio!r} at layer {layer}")


def page_bytes(config) -> int:
    return sum(shape[0] for _, shape in page_specs(config))


def continuation_layout(config, *, dspark: bool = True) -> StateLayout:
    """Per-request continuation state layout (SWA windows, C128 rings, partial pages, draft SWA)."""
    window_bytes = hca_latent_page_bytes(config.text["sliding_window"])
    regions = [
        (f"layer{layer}.window", (window_bytes,), torch.uint8)
        for layer in config.hca_layers
    ]
    for compressor in config.compressors:
        regions.extend(
            (f"{compressor.prefix}.{kind}", compressor.state_shape, torch.float32)
            for kind in ("values", "scores")
        )
    regions.extend((name, shape, torch.uint8) for name, shape in page_specs(config))
    if dspark:
        regions.extend(
            (f"draft{i}.window", (window_bytes,), torch.uint8)
            for i in range(config.draft.layers)
        )
    regions.extend(
        (
            ("last_hidden", (config.text["hidden_size"],), torch.bfloat16),
            ("consumed_tokens", (1,), torch.int64),
        )
    )
    return StateLayout.from_specs(regions)


@dataclass(frozen=True)
class ContinuationPool(PagedStateStorage):
    config: object

    @classmethod
    def allocate(
        cls,
        config,
        slots,
        pages,
        device,
        *,
        storage=None,
        maximum_pages=None,
        maximum_rows=None,
        backed_slots=None,
        dspark=True,
    ):
        backing = PagedStateStorage.allocate(
            continuation_layout(config, dspark=dspark),
            page_specs(config),
            slots,
            pages,
            device,
            storage=storage,
            maximum_pages=maximum_pages,
            maximum_rows=maximum_rows,
            backed_slots=backed_slots,
        )
        return cls(
            backing.backing,
            backing.layout,
            backing.memory_capacity,
            backing.pages,
            backing.reserved_backing,
            config,
        )

    def window(self, layer: int, *, draft: bool = False):
        return self.reserved_region(f"{'draft' if draft else 'layer'}{layer}.window")

    def dsa_latent_pages(self, layer: int):
        """View DSA latent pages as `[pages * 4, 64, 656]` uint8 for 64-token subpage indexing."""
        raw = self.pages[f"layer{layer}.latent_page"]
        return raw.view(-1, INDEX_SUBPAGE_TOKENS, DSA_RECORD_BYTES)

    def dsa_index_pages(self, layer: int):
        """View DSA index pages as `[pages * 4, 64 * 132]` uint8 for `PreparedFP8Indexer`."""
        raw = self.pages[f"layer{layer}.index_page"]
        return raw.view(-1, INDEX_SUBPAGE_TOKENS * INDEX_RECORD_BYTES)
