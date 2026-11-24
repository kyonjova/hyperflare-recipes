# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Hybrid ERNIE-5.5 attention: 37 HCA C128 layers + 6 DSA sparse MLA layers with VHA post-mix."""

from dataclasses import dataclass
from typing import NamedTuple

import torch
import triton
import triton.language as tl

from b12x.attention import sparse_mla
from b12x.attention._shared.mla.traits import ScaleFormat
from ignition.components.collectives import PreparedAllGather
from ignition.preparation.examples import declaration_tensor
from ignition.preparation.tuning import register_finalizer

from hyperflare_recipes.components.compressed_mla import (
    PreparedCompressedCacheWriter,
    PreparedCompressedMLA,
)
from hyperflare_recipes.components.fp8_indexer import (
    PreparedFP8Indexer,
    norm_and_write_fp8_index_keys,
    write_fp8_index_keys,
)
from hyperflare_recipes.components.gated_compression import PreparedGatedCompressor
from hyperflare_recipes.components.linear import QuantizedProjection, UnquantizedProjection
from hyperflare_recipes.components.rms_norm import PreparedRMSNorm
from hyperflare_recipes.components.rotary import PreparedRotaryAngles
from hyperflare_recipes.components.situ import sigmoid_gate
from hyperflare_recipes.components.sparse_mla import PreparedSparseMLA
from hyperflare_recipes.components.window_indices import PreparedWindowIndices

from .config import Compressor, HCA_LAYER_RATIO
from .pool import DSA_RECORD_BYTES, INDEX_SUBPAGE_TOKENS, PAGE_TOKENS, hca_latent_page_bytes

INDEX_ROWS = 2048
DSA_ATTN_CHUNK_ROWS = 512
SHARED_ROTARY_ROWS = 32
VHA_RANK = 16
VHA_CHUNK_ROWS = 256
DSA_TOPK = 2048
DSA_WINDOW = 128
DSA_MERGED_WIDTH = DSA_TOPK + DSA_WINDOW  # 2176
FLASHINDEXER_TOPK_HEADS = 8


@triton.jit
def _transfer_window(
    Source,
    Destination,
    StateRows,
    Starts,
    Counts,
    SOURCE_STRIDE: tl.constexpr,
    DEST_STRIDE: tl.constexpr,
    STAGED_TOKENS: tl.constexpr,
    COMMIT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    request, item = tl.program_id(0), tl.program_id(1)
    row = tl.load(StateRows + request)
    offset = tl.arange(0, BLOCK)
    if COMMIT:
        length = tl.load(Counts + request)
        token = tl.maximum(0, length - 128) + item
        live = (row > 0) & (token < length)
        source_row, source_token = request, 128 + token
        dest_row, dest_token = row, (tl.load(Starts + request) + token) % 128
        source_extent: tl.constexpr = STAGED_TOKENS
        dest_extent: tl.constexpr = 128
    else:
        live = row > 0
        source_row, source_token = row, item
        dest_row, dest_token = request, item
        source_extent: tl.constexpr = 128
        dest_extent: tl.constexpr = STAGED_TOKENS
    source_offset = tl.where(
        offset < 576,
        source_token * 576 + offset,
        source_extent * 576 + source_token * 8 + offset - 576,
    )
    dest_offset = tl.where(
        offset < 576,
        dest_token * 576 + offset,
        dest_extent * 576 + dest_token * 8 + offset - 576,
    )
    value = tl.load(
        Source + source_row * SOURCE_STRIDE + source_offset, live & (offset < 584), other=0
    )
    tl.store(
        Destination + dest_row * DEST_STRIDE + dest_offset, value, live & (offset < 584)
    )


def transfer_window(source, destination, metadata, staged_tokens, *, counts=None):
    torch.library.wrap_triton(_transfer_window)[(metadata.starts.numel(), 128)](
        source,
        destination,
        metadata.state_rows,
        metadata.starts,
        metadata.lengths if counts is None else counts,
        source.stride(0),
        destination.stride(0),
        staged_tokens,
        counts is not None,
        1024,
    )


@triton.jit
def _normalize_rotate(
    Query,
    Positions,
    Frequencies,
    ROW_STRIDE,
    HEAD_STRIDE,
    EPS,
    WIDTH: tl.constexpr,
    Angles=None,
    SHARED_ANGLES: tl.constexpr = False,
):
    """Per-head RMSNorm (BF16-rounded) followed by interleaved RoPE on the last 64 dims."""
    row, head = tl.program_id(0), tl.program_id(1)
    base = Query + row * ROW_STRIDE + head * HEAD_STRIDE
    offsets = tl.arange(0, WIDTH)
    value = tl.load(base + offsets).to(tl.float32)
    square = (value * value).to(tl.bfloat16).to(tl.float32)
    mean = (tl.sum(square, axis=0) / WIDTH).to(tl.bfloat16).to(tl.float32)
    scale = tl.rsqrt((mean + EPS).to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16).to(tl.float32)
    tl.store(base + offsets, (value * scale).to(tl.bfloat16), mask=offsets < WIDTH - 64)
    pair = tl.arange(0, 32)
    if SHARED_ANGLES:
        cos = tl.load(Angles + row * 64 + 2 * pair)
        sin = tl.load(Angles + row * 64 + 2 * pair + 1)
    else:
        angle = tl.load(Positions + row).to(tl.float32) * tl.load(Frequencies + pair)
        cos, sin = tl.cos(angle), tl.sin(angle)
    tail = base + WIDTH - 64 + 2 * pair
    a = (tl.load(tail).to(tl.float32) * scale).to(tl.bfloat16).to(tl.float32)
    b = (tl.load(tail + 1).to(tl.float32) * scale).to(tl.bfloat16).to(tl.float32)
    tl.store(tail, (a * cos - b * sin).to(tl.bfloat16))
    tl.store(tail + 1, (a * sin + b * cos).to(tl.bfloat16))


def normalize_rotate(query, positions, frequencies, eps, *, angles=None):
    torch.library.wrap_triton(_normalize_rotate)[(query.shape[0], query.shape[1])](
        query,
        positions,
        frequencies,
        query.stride(0),
        query.stride(1),
        eps,
        query.shape[2],
        angles,
        angles is not None,
    )


@triton.jit
def _partial_neox_rope_kernel(
    Values,
    Positions,
    Frequencies,
    ROW_STRIDE,
    HEAD_STRIDE,
    HALF_ROPE: tl.constexpr,
):
    """Apply NeoX split-half RoPE to the leading `2 * HALF_ROPE` dims of each head in place."""
    row = tl.program_id(0).to(tl.int64)
    head = tl.program_id(1).to(tl.int64)
    pos = tl.maximum(tl.load(Positions + row), 0).to(tl.float32)
    idx = tl.arange(0, HALF_ROPE)
    angle = pos * tl.load(Frequencies + idx)
    cos, sin = tl.cos(angle), tl.sin(angle)
    base = Values + row * ROW_STRIDE + head * HEAD_STRIDE
    x1 = tl.load(base + idx).to(tl.float32)
    x2 = tl.load(base + HALF_ROPE + idx).to(tl.float32)
    tl.store(base + idx, (x1 * cos - x2 * sin).to(Values.dtype.element_ty))
    tl.store(base + HALF_ROPE + idx, (x2 * cos + x1 * sin).to(Values.dtype.element_ty))


def partial_neox_rope(tensor: torch.Tensor, positions: torch.Tensor, frequencies: torch.Tensor) -> None:
    """In-place NeoX RoPE on the leading 64 dims of `[T, 128]` or `[T, H, 128]`."""
    if tensor.shape[0] == 0:
        return
    heads = tensor.shape[1] if tensor.ndim == 3 else 1
    head_stride = tensor.stride(1) if tensor.ndim == 3 else 0
    if tensor.is_cuda:
        torch.library.wrap_triton(_partial_neox_rope_kernel)[(tensor.shape[0], heads)](
            tensor,
            positions,
            frequencies,
            tensor.stride(0),
            head_stride,
            HALF_ROPE=32,
            num_warps=1,
        )
        return
    pos = positions.clamp_min(0).float()[:, None]
    angles = pos * frequencies.float()[None, :]
    cos, sin = angles.cos(), angles.sin()
    if tensor.ndim == 3:
        cos, sin = cos[:, None, :], sin[:, None, :]
    x1 = tensor[..., :32].float()
    x2 = tensor[..., 32:64].float()
    tensor[..., :32].copy_((x1 * cos - x2 * sin).to(tensor.dtype))
    tensor[..., 32:64].copy_((x2 * cos + x1 * sin).to(tensor.dtype))


@triton.jit
def _write_glm_nsa_latent_cache_kernel(
    KvNope,
    KPe,
    CacheBytes,
    Slots,
    kv_stride,
    kpe_stride,
    page_stride,
    PAGE_SIZE: tl.constexpr,
    RECORD_BYTES: tl.constexpr,
    FP8_MAX: tl.constexpr,
):
    """Write 656-byte GLM_NSA records: [0..512) FP8 E4M3 (4x128), [512..528) 4xFP32 scales, [528..656) BF16 k_pe."""
    row = tl.program_id(0).to(tl.int64)
    slot = tl.load(Slots + row).to(tl.int64)
    if slot < 0:
        return
    page = slot // PAGE_SIZE
    offset = slot % PAGE_SIZE
    rec_base = CacheBytes + page * page_stride + offset * RECORD_BYTES

    # Quantize four 128-element tiles into E4M3 with inline arbitrary FP32 scales.
    tile = tl.arange(0, 128)
    fp8_dst = rec_base.to(tl.pointer_type(tl.float8e4nv))
    scale_dst = (rec_base + 512).to(tl.pointer_type(tl.float32))
    for group in tl.static_range(4):
        cols = group * 128 + tile
        vals = tl.load(KvNope + row * kv_stride + cols).to(tl.float32)
        amax = tl.maximum(tl.max(tl.abs(vals), axis=0), 1e-12)
        scale = amax / FP8_MAX
        q = tl.clamp(vals / scale, -FP8_MAX, FP8_MAX).to(tl.float8e4nv)
        tl.store(fp8_dst + cols, q)
        tl.store(scale_dst + group, scale)

    # Copy 64 BF16 k_pe elements (128 bytes) at byte offset 528.
    rope_idx = tl.arange(0, 64)
    kpe = tl.load(KPe + row * kpe_stride + rope_idx)
    rope_dst = (rec_base + 528).to(tl.pointer_type(tl.bfloat16))
    tl.store(rope_dst + rope_idx, kpe)


def write_glm_nsa_latent_cache(
    kv_nope: torch.Tensor,
    k_pe: torch.Tensor,
    cache: torch.Tensor,
    slots: torch.Tensor,
) -> None:
    """Pack BF16 `kv_nope [T, 512]` and `k_pe [T, 64]` into `cache [subpages, 64, 656]` uint8."""
    rows = kv_nope.shape[0]
    if rows == 0:
        return
    torch.library.wrap_triton(_write_glm_nsa_latent_cache_kernel)[(rows,)](
        kv_nope,
        k_pe,
        cache,
        slots,
        kv_nope.stride(0),
        k_pe.stride(0),
        cache.stride(0),
        PAGE_SIZE=INDEX_SUBPAGE_TOKENS,
        RECORD_BYTES=DSA_RECORD_BYTES,
        FP8_MAX=448.0,
        num_warps=4,
    )


@triton.jit
def _merge_swa_topk_physical_kernel(
    TopkPtr,
    PageTablePtr,
    PositionsPtr,
    OutPtr,
    OutLenPtr,
    topk_stride,
    pt_stride,
    out_stride,
    MAX_PAGES: tl.constexpr,
    PAGE_TOKENS_C: tl.constexpr,
    WINDOW: tl.constexpr,
    TOPK: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Merge top-2048 indexed logical tokens + trailing 128 SWA tokens and map to physical token slots."""
    row = tl.program_id(0).to(tl.int64)
    pos = tl.load(PositionsPtr + row).to(tl.int32)
    if pos < 0:
        cols = tl.arange(0, BLOCK)
        for start in tl.static_range(0, WIDTH, BLOCK):
            c = start + cols
            tl.store(OutPtr + row * out_stride + c, -1, mask=c < WIDTH)
        tl.store(OutLenPtr + row, 0)
        return

    seq_len = pos + 1
    indexer_len = tl.minimum(tl.maximum(seq_len - WINDOW, 0), TOPK)
    swa_len = tl.minimum(seq_len, WINDOW)
    total_len = indexer_len + swa_len
    tl.store(OutLenPtr + row, total_len)

    cols = tl.arange(0, BLOCK)
    pt_row = PageTablePtr + row * pt_stride
    for start in tl.static_range(0, WIDTH, BLOCK):
        c = start + cols
        in_width = c < WIDTH
        is_topk = c < indexer_len
        is_swa = (~is_topk) & (c < total_len)
        topk_logical = tl.load(TopkPtr + row * topk_stride + c, mask=is_topk, other=-1)
        swa_logical = seq_len - swa_len + (c - indexer_len)
        logical = tl.where(is_topk, topk_logical, tl.where(is_swa, swa_logical, -1))
        valid = logical >= 0
        page_idx = tl.minimum(tl.maximum(logical // PAGE_TOKENS_C, 0), MAX_PAGES - 1)
        phys_page = tl.load(pt_row + page_idx, mask=valid, other=-1)
        phys_slot = tl.where(
            valid & (phys_page >= 0),
            phys_page * PAGE_TOKENS_C + (logical % PAGE_TOKENS_C),
            -1,
        )
        tl.store(OutPtr + row * out_stride + c, phys_slot, mask=in_width)


def merge_swa_topk_to_physical(
    topk_logical: torch.Tensor,
    token_page_table: torch.Tensor,
    positions: torch.Tensor,
    out_physical: torch.Tensor,
    out_lengths: torch.Tensor,
    *,
    window: int = DSA_WINDOW,
    topk: int = DSA_TOPK,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Combine SWA-excluded top-k logical indices with the last `window` tokens and translate to physical slots."""
    rows = positions.shape[0]
    width = topk + window
    if rows == 0:
        return out_physical, out_lengths
    if positions.is_cuda:
        torch.library.wrap_triton(_merge_swa_topk_physical_kernel)[(rows,)](
            topk_logical,
            token_page_table,
            positions,
            out_physical,
            out_lengths,
            topk_logical.stride(0),
            token_page_table.stride(0),
            out_physical.stride(0),
            MAX_PAGES=token_page_table.shape[1],
            PAGE_TOKENS_C=PAGE_TOKENS,
            WINDOW=window,
            TOPK=topk,
            WIDTH=width,
            BLOCK=256,
            num_warps=4,
        )
        return out_physical, out_lengths

    # CPU reference path for unit tests.
    pos = positions.to(torch.int64)
    live = pos >= 0
    seq_len = torch.where(live, pos + 1, 0)
    indexer_len = (seq_len - window).clamp(min=0, max=topk)
    swa_len = seq_len.clamp(max=window)
    total_len = indexer_len + swa_len
    out_lengths.copy_(total_len.to(torch.int32))
    cols = torch.arange(width, device=positions.device)[None, :]
    is_topk = cols < indexer_len[:, None]
    is_swa = (~is_topk) & (cols < total_len[:, None])
    topk_pad = torch.full((rows, width), -1, dtype=torch.int64, device=positions.device)
    topk_pad[:, :topk] = topk_logical.to(torch.int64)
    swa_logical = (seq_len - swa_len)[:, None] + (cols - indexer_len[:, None])
    logical = torch.where(is_topk, topk_pad, torch.where(is_swa, swa_logical, -1))
    valid = logical >= 0
    page_idx = (logical.clamp_min(0) // PAGE_TOKENS).clamp(max=token_page_table.shape[1] - 1)
    phys_page = token_page_table.to(torch.int64).gather(1, page_idx)
    phys_slot = torch.where(valid & (phys_page >= 0), phys_page * PAGE_TOKENS + (logical % PAGE_TOKENS), -1)
    out_physical.copy_(phys_slot.to(torch.int32))
    return out_physical, out_lengths


class PreparedVHAPostMix:
    """RoCE-safe TP low-rank head post-mixing: `o + V @ all_reduce(U^T @ o)` in FP32.

    Because RoCEnante `PreparedAllReduce` only supports BF16, TP>1 packs the
    FP32 `[chunk, 16, D]` partial as BF16 pairs `[chunk, 32 * D]`, gathers
    across ranks with `PreparedAllGather`, and sums the `world` FP32 halves
    locally. Prefill chunks at `VHA_CHUNK_ROWS = 256` to bound gather scratch.
    """

    def __init__(
        self,
        u: torch.Tensor,
        v: torch.Tensor,
        *,
        tokens: int,
        head_dim: int,
        world: int,
        gather_runtime,
        coordinator,
        session,
        name: str,
        examples=None,
    ):
        self.u = u
        self.v = v
        self.world = int(world)
        self.head_dim = int(head_dim)
        self.rank = u.shape[1]
        self.chunk_rows = min(tokens, VHA_CHUNK_ROWS)
        self.gather = None
        if self.world > 1:
            device = u.device
            # FP32 [chunk, 16, D] viewed as BF16 [chunk, 32 * D].
            local_cols = 2 * self.rank * self.head_dim
            def alloc(role, shape):
                if examples is not None:
                    return declaration_tensor(examples, role, shape, torch.bfloat16, device)
                return torch.empty(shape, dtype=torch.bfloat16, device=device)

            local_buf = alloc(f"vha.local.{self.head_dim}", (self.chunk_rows, local_cols))
            gathered_buf = alloc(
                f"vha.gathered.{self.head_dim}", (self.chunk_rows, self.world * local_cols)
            )
            self.gather = PreparedAllGather(
                local_buf,
                gathered_buf,
                runtime=gather_runtime,
                coordinator=coordinator,
                session=session,
                name=name + ".gather",
            ).functional(session=session)
            self.tail_rows = tokens % self.chunk_rows
            self.tail_gather = None
            if self.tail_rows:
                tail_local = alloc(f"vha.tail_local.{self.head_dim}", (self.tail_rows, local_cols))
                tail_gathered = alloc(
                    f"vha.tail_gathered.{self.head_dim}", (self.tail_rows, self.world * local_cols)
                )
                self.tail_gather = PreparedAllGather(
                    tail_local,
                    tail_gathered,
                    runtime=gather_runtime,
                    coordinator=coordinator,
                    session=session,
                    name=name + ".gather_tail",
                ).functional(session=session)

    def _mix_chunk(self, o: torch.Tensor, gather_op) -> torch.Tensor:
        rows = o.shape[0]
        u_t = self.u.float().t().expand(rows, -1, -1)
        z_local = torch.bmm(u_t, o.float())  # [rows, 16, D] FP32
        if gather_op is not None:
            flat_bf16 = z_local.contiguous().view(torch.bfloat16).view(rows, 2 * self.rank * self.head_dim)
            gathered = gather_op(flat_bf16).view(rows, self.world, self.rank, self.head_dim * 2)
            # Reinterpret BF16 pairs back to FP32 and sum across ranks.
            z = gathered.contiguous().view(torch.float32).view(rows, self.world, self.rank, self.head_dim).sum(dim=1)
        else:
            z = z_local
        v_f = self.v.float().expand(rows, -1, -1)
        return (o.float() + torch.bmm(v_f, z)).to(o.dtype)

    def __call__(self, o: torch.Tensor) -> torch.Tensor:
        rows = o.shape[0]
        if self.world == 1 or rows <= self.chunk_rows:
            return self._mix_chunk(o, self.gather)
        chunks = []
        for start in range(0, rows, self.chunk_rows):
            end = min(start + self.chunk_rows, rows)
            op = self.gather if (end - start) == self.chunk_rows else self.tail_gather
            chunks.append(self._mix_chunk(o[start:end], op))
        return torch.cat(chunks, dim=0)


@dataclass(frozen=True)
class AttentionScratch:
    hca_projected: torch.Tensor
    dsa_projected: torch.Tensor
    query_low: torch.Tensor
    hca_query: torch.Tensor
    window_values: torch.Tensor
    hca_attended: torch.Tensor
    hca_grouped: torch.Tensor
    dsa_query: torch.Tensor
    dsa_absorbed: torch.Tensor
    dsa_attended_latent: torch.Tensor
    dsa_gate: torch.Tensor
    output: torch.Tensor
    dense_indices: torch.Tensor
    index_query: torch.Tensor
    index_kw: torch.Tensor
    index_selected: torch.Tensor
    dsa_physical: torch.Tensor
    dsa_selected_lengths: torch.Tensor

    @classmethod
    def allocate(cls, tokens: int, world: int, device, *, blocks: int = 4096):
        def zeros(*shape, dtype=torch.bfloat16):
            return torch.zeros(shape, dtype=dtype, device=device)

        local_heads = 64 // world
        local_groups = 8 // world
        return cls(
            hca_projected=zeros(tokens, 1536),
            # q_a_proj (1024) + kv_a_proj_with_mqa (576, zero-padded to 640 for block-128 FP8).
            dsa_projected=zeros(tokens, 1664),
            query_low=zeros(tokens, 1024),
            hca_query=zeros(tokens, local_heads, 512),
            window_values=zeros(tokens, 512),
            hca_attended=zeros(tokens, local_heads, 512),
            hca_grouped=zeros(tokens, local_groups * 1024),
            dsa_query=zeros(tokens, local_heads, 256),
            dsa_absorbed=zeros(tokens, local_heads, 576),
            dsa_attended_latent=zeros(tokens, local_heads, 512),
            dsa_gate=zeros(tokens, local_heads * 256),
            output=zeros(tokens, 4096),
            dense_indices=zeros(tokens, blocks * (PAGE_TOKENS // HCA_LAYER_RATIO), dtype=torch.int32).fill_(-1),
            index_query=zeros(tokens, 64, 128),
            index_kw=zeros(tokens, 192),
            index_selected=zeros(tokens, DSA_TOPK, dtype=torch.int32).fill_(-1),
            dsa_physical=zeros(tokens, DSA_MERGED_WIDTH, dtype=torch.int32).fill_(-1),
            dsa_selected_lengths=zeros(tokens, dtype=torch.int32),
        )


class CompressionGeometry(NamedTuple):
    slots: torch.Tensor
    lengths: torch.Tensor
    index_lengths: torch.Tensor
    active_width: torch.Tensor
    position: torch.Tensor
    emitted: torch.Tensor
    emitted_slots: torch.Tensor


class AttentionGeometry(NamedTuple):
    window_slots: torch.Tensor
    window_indices: torch.Tensor
    window_lengths: torch.Tensor
    ratios: dict[int, CompressionGeometry]
    dense: torch.Tensor | None
    angles: dict[tuple[bool, int], torch.Tensor]
    dsa_slots: torch.Tensor | None
    dsa_cache_lengths: torch.Tensor | None
    dsa_indexer_lengths: torch.Tensor | None
    dsa_active_width: torch.Tensor | None
    dsa_subpage_table: torch.Tensor | None


class AttentionIndices(PreparedWindowIndices):
    """Shared window, C128 compression, rotary, and DSA slot/length geometry per invocation."""

    def __init__(self, metadata, *, mode: str, tokens: int, has_dsa: bool = True, examples=None):
        page_size = 128 if mode == "decode" else 128 + metadata.max_query_tokens
        width = page_size if mode == "draft" else 128
        super().__init__(metadata, mode=mode, tokens=tokens, page_size=page_size, width=width, visible=None)
        self.ratios = {}
        self.dense = None
        self.angles = {}
        self.has_dsa = bool(has_dsa)
        device = self.window_slots.device
        if self.has_dsa:
            self.dsa_slots = torch.zeros((tokens,), dtype=torch.int64, device=device)
            self.dsa_cache_lengths = torch.zeros((tokens,), dtype=torch.int32, device=device)
            self.dsa_indexer_lengths = torch.zeros((tokens,), dtype=torch.int32, device=device)
            self.dsa_active_width = torch.zeros((1,), dtype=torch.int32, device=device)
            # 256-token model pages expand to 4x 64-token subpages for FP8 indexer.
            subpage_shape = (tokens, metadata.page_table.shape[1] * (PAGE_TOKENS // INDEX_SUBPAGE_TOKENS))
            self.dsa_subpage_table = (
                declaration_tensor(examples, "attention.dsa_subpage_table", subpage_shape, torch.int32, device)
                if examples is not None
                else torch.zeros(subpage_shape, dtype=torch.int32, device=device)
            )
        else:
            self.dsa_slots = self.dsa_cache_lengths = self.dsa_indexer_lengths = None
            self.dsa_active_width = self.dsa_subpage_table = None

    def rotary_angles(self, rotary, *, compressed: bool, ratio: int = 0):
        key = compressed, ratio
        if key not in self.angles:
            positions = self.metadata.positions if ratio == 0 else self.compressed(ratio).position
            self.angles[key] = PreparedRotaryAngles(positions, rotary.frequencies[compressed])
        return self.angles[key]

    def compressed(self, ratio: int):
        if ratio not in self.ratios:
            device, tokens = self.window_slots.device, self.window_slots.shape[0]

            def zeros(dtype):
                return torch.zeros((tokens,), dtype=dtype, device=device)

            self.ratios[ratio] = CompressionGeometry(
                slots=zeros(torch.int64),
                lengths=zeros(torch.int32),
                index_lengths=zeros(torch.int32),
                active_width=torch.zeros(1, dtype=torch.int32, device=device),
                position=zeros(torch.int64),
                emitted=zeros(torch.bool),
                emitted_slots=zeros(torch.int64),
            )
        return self.ratios[ratio]

    def dense_pages(self, target):
        if self.dense is None:
            self.dense = target
        return self.dense

    def functional(self):
        self.compression_ratios = tuple(self.ratios)
        self.has_dense = self.dense is not None
        self.ratios = self.dense = None
        self.dsa_slots = self.dsa_cache_lengths = self.dsa_indexer_lengths = None
        self.dsa_active_width = self.dsa_subpage_table = None
        for operation in self.angles.values():
            operation.functional()
        super().functional()
        return self

    def __call__(self, metadata, visible=None):
        base = super().__call__(metadata, visible=visible)
        m = metadata
        live = m.positions >= 0
        position = m.positions.clamp_min(0)
        ratios = {}
        for ratio in self.compression_ratios:
            lengths = torch.where(live, (position + 1) // ratio, 0).to(torch.int32)
            pages = m.page_table[
                m.requests, (position // PAGE_TOKENS).clamp(max=m.page_table.shape[1] - 1)
            ]
            slots = torch.where(
                live, pages.long() * (PAGE_TOKENS // ratio) + (position % PAGE_TOKENS) // ratio, -1
            )
            emitted = live & ((m.positions + 1) % ratio == 0)
            ratios[ratio] = CompressionGeometry(
                slots=slots,
                lengths=lengths,
                index_lengths=lengths,
                active_width=lengths.amax().view(1),
                position=((position + 1) // ratio * ratio - ratio).clamp_min(0),
                emitted=emitted,
                emitted_slots=torch.where(emitted, slots, -1),
            )
        dense = None
        if self.has_dense:
            pages = m.token_page_table[:, :, None]
            records = PAGE_TOKENS // HCA_LAYER_RATIO
            dense = (
                torch.where(pages >= 0, pages * records + torch.arange(records, device=pages.device), -1)
                .flatten(1)
                .to(torch.int32)
            )
        angles = {
            key: operation(m.positions if key[1] == 0 else ratios[key[1]].position)
            for key, operation in self.angles.items()
        }
        dsa_slots = dsa_cache_lengths = dsa_indexer_lengths = dsa_active_width = dsa_subpage_table = None
        if self.has_dsa:
            pages = m.page_table[
                m.requests, (position // PAGE_TOKENS).clamp(max=m.page_table.shape[1] - 1)
            ]
            dsa_slots = torch.where(live, pages.long() * PAGE_TOKENS + (position % PAGE_TOKENS), -1)
            dsa_cache_lengths = torch.where(live, position + 1, 0).to(torch.int32)
            dsa_indexer_lengths = torch.where(live, (position + 1 - DSA_WINDOW).clamp_min(0), 0).to(torch.int32)
            dsa_active_width = dsa_indexer_lengths.amax().view(1)
            subpages_per_page = PAGE_TOKENS // INDEX_SUBPAGE_TOKENS
            pt = m.token_page_table[:, :, None]
            offsets = torch.arange(subpages_per_page, device=pt.device, dtype=pt.dtype)
            dsa_subpage_table = torch.where(pt >= 0, pt * subpages_per_page + offsets, -1).flatten(1).to(torch.int32)
        return AttentionGeometry(
            window_slots=base.window_slots,
            window_indices=base.window_indices,
            window_lengths=base.window_lengths,
            ratios=ratios,
            dense=dense,
            angles=angles,
            dsa_slots=dsa_slots,
            dsa_cache_lengths=dsa_cache_lengths,
            dsa_indexer_lengths=dsa_indexer_lengths,
            dsa_active_width=dsa_active_width,
            dsa_subpage_table=dsa_subpage_table,
        )


class PreparedHCAAttention:
    """ERNIE-5.5 HCA layer (37 target C128 layers + 3 SWA-only DSpark draft layers)."""

    def __init__(
        self,
        layer: int,
        weights: dict,
        source: torch.Tensor,
        scratch: AttentionScratch,
        metadata,
        pool,
        rotary,
        *,
        mode: str,
        world: int,
        vha_gather_runtime,
        coordinator,
        session,
        name: str,
        workspace,
        indices=None,
        examples=None,
    ):
        self.pool = pool
        self.heads = scratch.hca_query.shape[1]
        self.rotary = rotary
        self.mode = mode
        self.ratio = 0 if mode == "draft" else HCA_LAYER_RATIO
        self.eps = pool.config.text["rms_norm_eps"]
        self.local_groups = 8 // world
        self.o_lora_rank = pool.config.text["o_lora_rank"]
        tokens, rows, device = source.shape[0], metadata.starts.numel(), source.device
        self.window = pool.window(layer, draft=mode == "draft")
        self.owns_indices = indices is None
        if indices is None:
            indices = AttentionIndices(metadata, mode=mode, tokens=tokens, has_dsa=False)
        self.indices = indices
        self.window_page_size = indices.window_page_size
        self.window_shape = (rows, hca_latent_page_bytes(self.window_page_size))
        self.window_cache = (
            self.window
            if mode == "decode"
            else declaration_tensor(
                examples, "attention.window", self.window_shape, torch.uint8, device
            )
        )
        shared = indices.compressed(self.ratio) if self.ratio else None
        share_angles = tokens <= SHARED_ROTARY_ROWS
        self.angles = indices.rotary_angles(rotary, compressed=bool(self.ratio)) if share_angles else None
        self.compressed_angles = (
            indices.rotary_angles(rotary, compressed=True, ratio=self.ratio)
            if self.ratio and share_angles
            else None
        )

        self.qkv = QuantizedProjection(
            source,
            weights["qkv_a"],
            scratch.hca_projected,
            session=session,
            name=name + ".qkv_a",
            workspace=workspace,
            activation_mode="auto",
            examples=examples,
            example_role="attention.hca_qkv",
        ).functional(session=session)
        self.qnorm = PreparedRMSNorm(
            scratch.hca_projected[:, :1024],
            weights["q_norm"],
            scratch.query_low,
            eps=self.eps,
            session=session,
            name=name + ".q_norm",
        ).functional()
        self.kvnorm = PreparedRMSNorm(
            scratch.hca_projected[:, 1024:],
            weights["kv_norm"],
            scratch.window_values,
            eps=self.eps,
            session=session,
            name=name + ".kv_norm",
        ).functional()
        self.qb = QuantizedProjection(
            scratch.query_low,
            weights["q_b"],
            scratch.hca_query.view(tokens, -1),
            session=session,
            name=name + ".q_b",
            workspace=workspace,
            activation_mode="auto",
            examples=examples,
            example_role="attention.hca_query",
        ).functional(session=session)
        self.o_gate = QuantizedProjection(
            scratch.query_low,
            weights["o_gate"],
            scratch.hca_grouped,
            session=session,
            name=name + ".o_gate",
            workspace=workspace,
            activation_mode="auto",
            examples=examples,
            example_role="attention.hca_gate",
        ).functional(session=session)

        o_group_in = (
            declaration_tensor(examples, "attention.hca_o_group_in", (tokens, 4096), torch.bfloat16, device)
            if examples is not None
            else torch.empty((tokens, 4096), dtype=torch.bfloat16, device=device)
        )
        o_group_out = (
            declaration_tensor(
                examples, "attention.hca_o_group_out", (tokens, self.o_lora_rank), torch.bfloat16, device
            )
            if examples is not None
            else torch.empty((tokens, self.o_lora_rank), dtype=torch.bfloat16, device=device)
        )
        self.o_group = tuple(
            QuantizedProjection(
                o_group_in,
                weights["o_group"][g],
                o_group_out,
                session=session,
                name=f"{name}.o_group{g}",
                workspace=workspace,
                activation_mode="auto",
                examples=examples,
                example_role="attention.hca_o_group",
            ).functional(session=session)
            for g in range(self.local_groups)
        )
        self.o_proj = QuantizedProjection(
            scratch.hca_grouped,
            weights["o_proj"],
            scratch.output,
            session=session,
            name=name + ".o_proj",
            workspace=workspace,
            activation_mode="auto",
            examples=examples,
            example_role="attention.hca_out",
        ).functional(session=session)

        self.vha = PreparedVHAPostMix(
            weights["vha_u"],
            weights["vha_v"],
            tokens=tokens,
            head_dim=512,
            world=world,
            gather_runtime=vha_gather_runtime,
            coordinator=coordinator,
            session=session,
            name=name + ".vha",
            examples=examples,
        )

        self.window_writer = PreparedCompressedCacheWriter(
            scratch.window_values,
            self.window_cache,
            indices.window_slots,
            page_size=self.window_page_size,
            kind="swa",
            session=session,
            name=name + ".window_write",
            cache_format="deepseek_v4",
        ).functional(session=session, cache=None if mode != "decode" else self.window)

        self.compressor = self.compressed_writer = self.compression = None
        indexed = {}
        if self.ratio:
            spec = Compressor(layer, self.ratio, 512)
            width = spec.state_shape[1]
            compressed_buf = declaration_tensor(
                examples, "attention.compression", (tokens, 2 * width), torch.float32, device
            )
            self.compression = UnquantizedProjection(
                source,
                weights["compression"],
                compressed_buf,
                session=session,
                name=name + ".compression",
            ).functional(session=session)
            self.compressor = PreparedGatedCompressor(
                spec,
                weights["compressor"],
                source,
                metadata,
                pool,
                shared.slots,
                session=session,
                name=name + ".compress",
                deferred=mode == "verify",
                examples=examples,
                example_role="attention.compressor",
                emitted=shared.emitted,
                emitted_slots=shared.emitted_slots,
                projected=(compressed_buf[:, :width], compressed_buf[:, width : 2 * width]),
            ).functional(session=session)
            self.compressed_writer = PreparedCompressedCacheWriter(
                self.compressor.output,
                pool.pages[f"layer{layer}.latent_page"],
                self.compressor.slots,
                page_size=PAGE_TOKENS // self.ratio,
                kind="indexed",
                session=session,
                name=name + ".compressed_write",
                cache_format="deepseek_v4",
            ).functional(session=session)
            selected = indices.dense_pages(
                scratch.dense_indices[:, : metadata.page_table.shape[1] * (PAGE_TOKENS // self.ratio)]
            )
            indexed = {
                "indexed": pool.pages[f"layer{layer}.latent_page"],
                "indexed_indices": selected,
                "indexed_lengths": shared.index_lengths,
                "indexed_page_size": PAGE_TOKENS // self.ratio,
            }

        self.attention = PreparedCompressedMLA(
            scratch.hca_query,
            self.window_cache,
            indices.window_indices,
            indices.window_lengths,
            weights["sink"],
            scratch.hca_attended,
            window_page_size=self.window_page_size,
            mode="decode" if mode in ("decode", "verify") or (mode == "draft" and tokens <= 32) else "extend",
            session=session,
            name=name + ".attention",
            workspace=workspace,
            cache_format="deepseek_v4",
            **indexed,
        ).functional(session=session, window=None if mode != "decode" else self.window)

        register_finalizer(session, self._release_examples)
        if self.owns_indices:
            self.indices.functional()

    def _release_examples(self):
        self.window_cache = None

    def rotate(self, tensor, metadata, indices, *, compression: bool = False, inverse: bool = False):
        operation = self.compressed_angles if compression else self.angles
        if operation is not None:
            key = (True, self.ratio) if compression else (bool(self.ratio), 0)
            operation.apply(tensor, indices.angles[key], inverse=inverse)
        else:
            position = (
                indices.ratios[self.ratio].position if compression else metadata.positions.clamp_min(0)
            )
            self.rotary.apply(
                tensor, position, compressed=compression or bool(self.ratio), inverse=inverse
            )

    def __call__(self, source: torch.Tensor, metadata, indices=None):
        m = metadata
        if self.owns_indices:
            indices = self.indices(m)
        window = (
            self.window
            if self.mode == "decode"
            else torch.empty(self.window_shape, dtype=torch.uint8, device=source.device)
        )
        if self.mode != "decode":
            transfer_window(self.window, window, m, self.window_page_size)

        shared = indices.ratios[self.ratio] if self.ratio else None
        projected = self.qkv(source)
        query_low = self.qnorm(projected[:, :1024])
        gate = self.o_gate(query_low)
        window_values = self.kvnorm(projected[:, 1024:])
        query = self.qb(query_low).view(source.shape[0], self.heads, 512)

        angles = None if self.angles is None else indices.angles[(bool(self.ratio), 0)]
        normalize_rotate(
            query,
            m.positions if angles is not None else m.positions.clamp_min(0),
            self.rotary.frequencies[bool(self.ratio)],
            self.eps,
            angles=angles,
        )
        self.rotate(window_values, m, indices)
        self.window_writer(window_values, indices.window_slots, cache=window)

        compressed_records = None
        selected = indices.dense if self.ratio else None
        if self.ratio:
            comp_proj = self.compression(source)
            width = self.compressor.spec.state_shape[1]
            latent, values, scores, _, slots = self.compressor(
                source,
                m,
                shared.slots,
                projected=(comp_proj[:, :width], comp_proj[:, width : 2 * width]),
                emitted=shared.emitted,
                emitted_slots=shared.emitted_slots,
            )
            compressed_records = (values, scores)
            self.rotate(latent, m, indices, compression=True)
            self.compressed_writer(latent, slots)

        attended = self.attention(
            query,
            indices.window_indices,
            indices.window_lengths,
            indexed_indices=selected,
            indexed_lengths=None if shared is None else shared.index_lengths,
            window=window,
        )
        attended = self.vha(attended)
        self.rotate(attended, m, indices, inverse=True)

        grouped_in = attended.view(source.shape[0], self.local_groups, 4096)
        grouped = torch.cat(
            [proj(grouped_in[:, g, :].contiguous()) for g, proj in enumerate(self.o_group)],
            dim=-1,
        )
        gated = sigmoid_gate(grouped, gate)
        output = self.o_proj(gated)
        records = (window, compressed_records)
        if self.mode == "prefill":
            self.commit(m.lengths, records, m)
        return output, records

    def commit(self, counts, records, metadata):
        window, compressed = records
        transfer_window(window, self.window, metadata, self.window_page_size, counts=counts)
        if self.mode == "verify" and self.compressor is not None and compressed is not None:
            self.compressor.commit(counts, compressed, metadata)


class PreparedDSAAttention:
    """ERNIE-5.5 DSA layer (layers 7, 14, 21, 28, 35, 42): full-token MLA + FP8 indexer + VHA + gate."""

    def __init__(
        self,
        layer: int,
        weights: dict,
        source: torch.Tensor,
        scratch: AttentionScratch,
        metadata,
        pool,
        rotary,
        *,
        mode: str,
        world: int,
        vha_gather_runtime,
        coordinator,
        session,
        name: str,
        workspace,
        indices: AttentionIndices,
        examples=None,
    ):
        self.pool = pool
        self.layer = layer
        self.mode = mode
        self.heads = scratch.dsa_query.shape[1]
        self.rotary = rotary
        self.eps = pool.config.text["rms_norm_eps"]
        self.w_uk = weights["w_uk"]  # [H_local, 192, 512] BF16
        self.w_uv = weights["w_uv"]  # [H_local, 512, 256] BF16
        self.k_norm_weight = weights["index_k_norm_weight"]
        self.k_norm_bias = weights["index_k_norm_bias"]
        self.latent_cache = pool.dsa_latent_pages(layer)
        self.index_cache = pool.dsa_index_pages(layer)

        tokens, device = source.shape[0], source.device
        self.qkv = QuantizedProjection(
            source,
            weights["qkv_a"],
            scratch.dsa_projected,
            session=session,
            name=name + ".qkv_a",
            workspace=workspace,
            activation_mode="auto",
            examples=examples,
            example_role="attention.dsa_qkv",
        ).functional(session=session)
        self.qnorm = PreparedRMSNorm(
            scratch.dsa_projected[:, :1024],
            weights["q_norm"],
            scratch.query_low,
            eps=self.eps,
            session=session,
            name=name + ".q_norm",
        ).functional()
        self.kvnorm = PreparedRMSNorm(
            scratch.dsa_projected[:, 1024:1536],
            weights["kv_norm"],
            scratch.window_values,
            eps=self.eps,
            session=session,
            name=name + ".kv_norm",
        ).functional()
        self.qb = QuantizedProjection(
            scratch.query_low,
            weights["q_b"],
            scratch.dsa_query.view(tokens, -1),
            session=session,
            name=name + ".q_b",
            workspace=workspace,
            activation_mode="auto",
            examples=examples,
            example_role="attention.dsa_query",
        ).functional(session=session)
        self.o_gate = QuantizedProjection(
            scratch.query_low,
            weights["o_gate"],
            scratch.dsa_gate,
            session=session,
            name=name + ".o_gate",
            workspace=workspace,
            activation_mode="auto",
            examples=examples,
            example_role="attention.dsa_gate",
        ).functional(session=session)
        self.o_proj = QuantizedProjection(
            scratch.dsa_gate,
            weights["o_proj"],
            scratch.output,
            session=session,
            name=name + ".o_proj",
            workspace=workspace,
            activation_mode="auto",
            examples=examples,
            example_role="attention.dsa_out",
        ).functional(session=session)

        self.index_q = QuantizedProjection(
            scratch.query_low,
            weights["index_q"],
            scratch.index_query.view(tokens, -1),
            session=session,
            name=name + ".index_q",
            workspace=workspace,
            activation_mode="auto",
            examples=examples,
            example_role="attention.dsa_index_q",
        ).functional(session=session)
        self.index_kw = UnquantizedProjection(
            source,
            weights["index_kw"],
            scratch.index_kw,
            session=session,
            name=name + ".index_kw",
        ).functional(session=session)

        # Optional FlashIndexer top-8 head gate in prefill.
        self.flashindexer_gate = None
        self.flashindexer_bias = None
        self.use_flashindexer = (
            mode == "prefill"
            and "flashindexer_gate" in weights
            and "flashindexer_bias" in weights
        )
        if self.use_flashindexer:
            gate_out = declaration_tensor(
                examples, "attention.flashindexer_gates", (tokens, 64), torch.bfloat16, device
            )
            self.flashindexer_gate = UnquantizedProjection(
                source,
                weights["flashindexer_gate"],
                gate_out,
                session=session,
                name=name + ".flashindexer_gate",
            ).functional(session=session)
            self.flashindexer_bias = weights["flashindexer_bias"]

        idx_heads = FLASHINDEXER_TOPK_HEADS if self.use_flashindexer else 64
        idx_q_example = (
            declaration_tensor(
                examples, "attention.dsa_fi_query", (tokens, idx_heads, 128), torch.bfloat16, device
            )
            if self.use_flashindexer
            else scratch.index_query
        )
        idx_w_example = (
            declaration_tensor(
                examples, "attention.dsa_fi_weights", (tokens, idx_heads), torch.bfloat16, device
            )
            if self.use_flashindexer
            else scratch.index_kw[:, 128:]
        )
        self.selectors = []
        for first in range(0, tokens, INDEX_ROWS):
            last = min(tokens, first + INDEX_ROWS)
            self.selectors.append(
                PreparedFP8Indexer(
                    idx_q_example[first:last],
                    idx_w_example[first:last],
                    self.index_cache,
                    indices.dsa_subpage_table[first:last],
                    indices.dsa_indexer_lengths[first:last],
                    indices.dsa_active_width,
                    scratch.index_selected[first:last],
                    mode="prefill" if mode == "prefill" else "decode",
                    session=session,
                    name=f"{name}.index{first}",
                    workspace=workspace,
                    topk=DSA_TOPK,
                    head_weight_scale=64**-0.5 if self.use_flashindexer else None,
                    hadamard=True,
                    key_power_of_two=True,
                    # On SM120/SM121 (99 KiB max SMEM), DSAFusedIndexerKernel with 64 heads + topk=2048
                    # allocates 106 KiB SMEM; paged_tiled fits cleanly and already serves B>=8.
                    route="paged_tiled" if mode != "prefill" else "auto",
                    prefill_records_per_call=32768,
                    examples=examples,
                    example_role=f"attention.dsa_indexer_{idx_heads}",
                ).functional(session=session)
            )

        # Bound sparse-MLA split-KV scratch by chunking prefill rows and capping chunks per row:
        # 2176 / 64 = 34 max KV tiles; 512-row prefill chunks with 4 splits saturate GB10 in ~64 MiB.
        self.attentions = []
        for first in range(0, tokens, DSA_ATTN_CHUNK_ROWS):
            last = min(tokens, first + DSA_ATTN_CHUNK_ROWS)
            chunk_rows = last - first
            max_chunks = 34 if chunk_rows <= 32 else (16 if chunk_rows <= 128 else 4)
            caps = sparse_mla.Caps(
                device=device,
                num_q_heads=self.heads,
                max_q_rows=chunk_rows,
                max_width=DSA_MERGED_WIDTH,
                softmax_scale=256**-0.5,
                dtype=torch.bfloat16,
                kv_dtype=torch.uint8,
                head_dim=576,
                v_head_dim=512,
                model_type=int(sparse_mla.ModelType.GLM_NSA),
                scale_format=int(ScaleFormat.ARBITRARY_FP32),
                mode="decode",
                page_size=INDEX_SUBPAGE_TOKENS,
                max_chunks_per_row=max_chunks,
                has_attention_sink=True,
            )
            self.attentions.append(
                PreparedSparseMLA(
                    scratch.dsa_absorbed[first:last],
                    self.latent_cache,
                    scratch.dsa_physical[first:last],
                    indices.dsa_cache_lengths[first:last],
                    scratch.dsa_selected_lengths[first:last],
                    scratch.dsa_attended_latent[first:last],
                    caps=caps,
                    session=session,
                    name=f"{name}.sparse_mla{first}",
                    workspace=workspace,
                    attention_sink=weights["sink"],
                ).functional(session=session)
            )

        self.vha = PreparedVHAPostMix(
            weights["vha_u"],
            weights["vha_v"],
            tokens=tokens,
            head_dim=256,
            world=world,
            gather_runtime=vha_gather_runtime,
            coordinator=coordinator,
            session=session,
            name=name + ".vha",
            examples=examples,
        )

    def __call__(self, source: torch.Tensor, metadata, indices: AttentionGeometry):
        tokens = source.shape[0]
        projected = self.qkv(source)
        query_low = self.qnorm(projected[:, :1024])
        gate = self.o_gate(query_low)
        kv_nope = self.kvnorm(projected[:, 1024:1536])
        k_pe = projected[:, 1536:1600].contiguous()
        q = self.qb(query_low).view(tokens, self.heads, 256)

        # Interleaved default RoPE (theta=8000000) on trailing 64 dims of q and on k_pe.
        pos = metadata.positions.clamp_min(0)
        self.rotary.apply(q, pos, compressed=False)
        self.rotary.apply(k_pe, pos, compressed=False)

        # Pack [kv_nope, k_pe] into 656-byte GLM_NSA cache records.
        write_glm_nsa_latent_cache(kv_nope, k_pe, self.latent_cache, indices.dsa_slots)

        # Indexer projections, k LayerNorm, leading-64 NeoX partial RoPE, and FP8 key write.
        idx_q = self.index_q(query_low).view(tokens, 64, 128)
        idx_kw = self.index_kw(source)
        idx_k = torch.empty((tokens, 128), dtype=torch.bfloat16, device=source.device)
        # Apply FP32 LayerNorm on k, then NeoX partial RoPE on leading 64 dims, then write keys.
        norm_and_write_fp8_index_keys(
            idx_kw[:, :128].contiguous(),
            self.k_norm_weight,
            self.k_norm_bias,
            1e-6,
            torch.full_like(indices.dsa_slots, -1),
            self.index_cache,
            hadamard=True,
            power_of_two=True,
            out=idx_k,
        )
        partial_neox_rope(idx_q, pos, self.rotary.frequencies[False])
        partial_neox_rope(idx_k, pos, self.rotary.frequencies[False])
        write_fp8_index_keys(
            idx_k, indices.dsa_slots, self.index_cache, hadamard=True, power_of_two=True
        )
        idx_w = idx_kw[:, 128:].contiguous()

        if self.use_flashindexer:
            gates = self.flashindexer_gate(source).float()
            top_heads = (gates + self.flashindexer_bias).topk(FLASHINDEXER_TOPK_HEADS, dim=-1).indices
            idx_q = torch.gather(idx_q, 1, top_heads[:, :, None].expand(-1, -1, 128)).contiguous()
            idx_w = (torch.gather(idx_w.float(), 1, top_heads) * torch.gather(gates, 1, top_heads)).to(
                torch.bfloat16
            ).contiguous()

        selected_parts = [
            selector(
                idx_q[first : first + INDEX_ROWS],
                idx_w[first : first + INDEX_ROWS],
                indices.dsa_subpage_table[first : first + INDEX_ROWS],
                indices.dsa_indexer_lengths[first : first + INDEX_ROWS],
                indices.dsa_active_width,
            )
            for first, selector in zip(range(0, tokens, INDEX_ROWS), self.selectors, strict=True)
        ]
        topk_logical = selected_parts[0] if len(selected_parts) == 1 else torch.cat(selected_parts, dim=0)

        physical = torch.empty((tokens, DSA_MERGED_WIDTH), dtype=torch.int32, device=source.device)
        selected_lengths = torch.empty((tokens,), dtype=torch.int32, device=source.device)
        merge_swa_topk_to_physical(
            topk_logical, metadata.token_page_table, metadata.positions, physical, selected_lengths
        )

        # Absorb q_nope [T, H_local, 192] via W_uk [H_local, 192, 512] -> [T, H_local, 512] + q_pe [64].
        q_nope_abs = torch.bmm(q[:, :, :192].transpose(0, 1), self.w_uk).transpose(0, 1)
        q_absorbed = torch.cat((q_nope_abs, q[:, :, 192:]), dim=-1).contiguous()

        if len(self.attentions) == 1:
            attended_latent = self.attentions[0](
                q_absorbed, physical, indices.dsa_cache_lengths, selected_lengths
            )
        else:
            attended_latent = torch.cat(
                [
                    attn(
                        q_absorbed[first : first + DSA_ATTN_CHUNK_ROWS],
                        physical[first : first + DSA_ATTN_CHUNK_ROWS],
                        indices.dsa_cache_lengths[first : first + DSA_ATTN_CHUNK_ROWS],
                        selected_lengths[first : first + DSA_ATTN_CHUNK_ROWS],
                    )
                    for first, attn in zip(
                        range(0, tokens, DSA_ATTN_CHUNK_ROWS), self.attentions, strict=True
                    )
                ],
                dim=0,
            )
        # Expand latent [T, H_local, 512] via W_uv [H_local, 512, 256] -> [T, H_local, 256].
        attended = torch.bmm(attended_latent.transpose(0, 1), self.w_uv).transpose(0, 1).contiguous()
        attended = self.vha(attended)
        gated = sigmoid_gate(attended.reshape(tokens, -1), gate)
        output = self.o_proj(gated)
        return output, None

    def commit(self, counts, records, metadata):
        # DSA writes directly to paged latent/index caches; no continuation ring to commit.
        del counts, records, metadata
