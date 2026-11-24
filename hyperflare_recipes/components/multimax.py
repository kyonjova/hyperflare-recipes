# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Multimax piecewise-polynomial logit transform and DSpark selector adapter."""

import torch
import triton
import triton.language as tl


@triton.jit
def _multimax_kernel(
    logits_ptr,
    ranges_ptr,
    ts_ptr,
    rows,
    cols,
    row_stride,
    BLOCK_C: tl.constexpr,
    PRESERVE_NEG_INF: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    col_block = tl.program_id(1).to(tl.int64)
    col = col_block * BLOCK_C + tl.arange(0, BLOCK_C)
    mask = (row < rows) & (col < cols)
    r0 = tl.load(ranges_ptr + 0).to(tl.float32)
    r1 = tl.load(ranges_ptr + 1).to(tl.float32)
    r2 = tl.load(ranges_ptr + 2).to(tl.float32)
    r3 = tl.load(ranges_ptr + 3).to(tl.float32)
    t0 = tl.load(ts_ptr + 0).to(tl.float32)
    t1 = tl.load(ts_ptr + 1).to(tl.float32)
    t2 = tl.load(ts_ptr + 2).to(tl.float32)
    t3 = tl.load(ts_ptr + 3).to(tl.float32)

    offset = row * row_stride + col
    x = tl.load(logits_ptr + offset, mask=mask, other=0.0).to(tl.float32)
    d0 = tl.maximum(r0 - x, 0.0)
    d1 = tl.maximum(x - r1, 0.0)
    d2 = tl.maximum(r2 - x, 0.0)
    d3 = tl.maximum(x - r3, 0.0)
    out = x + t0 * d0
    out = out + t1 * d1
    out = out + t2 * (d2 * d2)
    out = out + t3 * (d3 * d3)
    if PRESERVE_NEG_INF:
        out = tl.where(x == float("-inf"), x, out)
    tl.store(logits_ptr + offset, out, mask=mask)


def apply_multimax(
    logits: torch.Tensor,
    ranges: torch.Tensor,
    ts: torch.Tensor,
    *,
    preserve_neg_inf: bool = False,
    inplace: bool = True,
) -> torch.Tensor:
    """Apply ERNIE-5.5 multimax transform to 2D logits `[rows, vocab]`.

    Uses a Triton kernel with `enable_fp_fusion=False` on CUDA FP32 tensors to
    match eager FP32 operation ordering. When `preserve_neg_inf=True`, masked
    `-inf` logits remain `-inf` instead of turning into `NaN` if `t[0]` or
    `t[2]` is positive.
    """
    if logits.ndim != 2:
        raise ValueError(f"multimax requires 2D logits, got shape {tuple(logits.shape)}")
    if ranges.shape != (4,) or ts.shape != (4,):
        raise ValueError("multimax requires 4-element ranges and ts vectors")
    rows, cols = logits.shape
    if rows == 0 or cols == 0:
        return logits if inplace else logits.clone()

    if logits.is_cuda and logits.dtype == torch.float32 and logits.stride(1) == 1:
        target = logits if inplace else logits.clone()
        block_c = 1024
        grid = (rows, triton.cdiv(cols, block_c))
        torch.library.wrap_triton(_multimax_kernel)[grid](
            target,
            ranges.contiguous(),
            ts.contiguous(),
            rows,
            cols,
            target.stride(0),
            BLOCK_C=block_c,
            PRESERVE_NEG_INF=preserve_neg_inf,
            num_warps=4,
            enable_fp_fusion=False,
        )
        return target

    orig_dtype = logits.dtype
    x = logits.float() if orig_dtype != torch.float32 else logits
    r = ranges.to(device=x.device, dtype=torch.float32)
    t = ts.to(device=x.device, dtype=torch.float32)
    out = (
        x
        + t[0] * (r[0] - x).relu()
        + t[1] * (x - r[1]).relu()
        + t[2] * (r[2] - x).relu().square()
        + t[3] * (x - r[3]).relu().square()
    )
    if preserve_neg_inf:
        out = torch.where(x == float("-inf"), x, out)
    out = out.to(orig_dtype)
    if inplace:
        logits.copy_(out)
        return logits
    return out


class MultimaxSelector:
    """Adapter exposing ERNIE-5.5 multimax to `PreparedParallelDraftRound`.

    Attached as `draft.selector`, this runs after Markov transition bias and
    preserves `-inf` masked entries while disabling fused shortcuts that bypass
    post-Markov logit transforms.
    """

    def __init__(self, ranges: torch.Tensor, ts: torch.Tensor):
        if ranges.shape != (4,) or ts.shape != (4,):
            raise ValueError("MultimaxSelector requires 4-element ranges and ts vectors")
        self.ranges = ranges
        self.ts = ts

    def apply(self, logits, markov_token, position, selector_features):
        del markov_token, position, selector_features
        return apply_multimax(logits, self.ranges, self.ts, preserve_neg_inf=True, inplace=True)
