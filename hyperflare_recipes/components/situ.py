# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Prepared SiTU MLPs and sigmoid-gated output helpers."""

import torch
import triton
import triton.language as tl

from hyperflare_recipes.components.linear import QuantizedProjection, UnquantizedProjection
from hyperflare_recipes.components.weights import MLPWeights


@triton.jit
def _situ_kernel(
    gate_up_ptr,
    out_ptr,
    rows,
    width,
    in_row_stride,
    out_row_stride,
    beta,
    linear_beta,
    BLOCK_W: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    col_block = tl.program_id(1).to(tl.int64)
    col = col_block * BLOCK_W + tl.arange(0, BLOCK_W)
    mask = (row < rows) & (col < width)
    in_base = row * in_row_stride + col
    gate = tl.load(gate_up_ptr + in_base, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(gate_up_ptr + in_base + width, mask=mask, other=0.0).to(tl.float32)
    sig = tl.sigmoid(gate)
    # tl.extra.cuda.libdevice.tanh is exact FP32 tanh on CUDA; compute via exp
    # for portable Triton compilation: tanh(z) = (e^(2z) - 1) / (e^(2z) + 1)
    # = 2 * sigmoid(2z) - 1, which avoids overflow for all finite z.
    tanh_gate = 2.0 * tl.sigmoid(2.0 * (gate / beta)) - 1.0
    tanh_up = 2.0 * tl.sigmoid(2.0 * (up / linear_beta)) - 1.0
    activated = (beta * tanh_gate * sig) * (linear_beta * tanh_up)
    tl.store(out_ptr + row * out_row_stride + col, activated.to(tl.bfloat16), mask=mask)


def situ(
    gate_up: torch.Tensor,
    out: torch.Tensor | None = None,
    *,
    beta: float = 4.0,
    linear_beta: float = 25.0,
) -> torch.Tensor:
    """Apply SiTU activation `beta * tanh(gate/beta) * sigmoid(gate) * linear_beta * tanh(up/linear_beta)`.

    `gate_up` has shape `[rows, 2 * width]` with gate in `[:, :width]` and up in
    `[:, width:]`, matching `PreparedMLP`'s gate-then-up concatenation layout.
    """
    if gate_up.ndim != 2 or gate_up.shape[1] % 2 != 0:
        raise ValueError(f"SiTU requires 2D [rows, 2*width] input, got {tuple(gate_up.shape)}")
    rows, double_width = gate_up.shape
    width = double_width // 2
    if out is None:
        out = torch.empty((rows, width), dtype=gate_up.dtype, device=gate_up.device)
    elif out.shape != (rows, width) or out.dtype != gate_up.dtype or out.device != gate_up.device:
        raise ValueError("SiTU output buffer must match input [rows, width], dtype, and device")
    if rows == 0 or width == 0:
        return out
    if gate_up.is_cuda and gate_up.dtype == torch.bfloat16 and gate_up.stride(1) == 1 and out.stride(1) == 1:
        block_w = 256
        grid = (rows, triton.cdiv(width, block_w))
        torch.library.wrap_triton(_situ_kernel)[grid](
            gate_up,
            out,
            rows,
            width,
            gate_up.stride(0),
            out.stride(0),
            float(beta),
            float(linear_beta),
            BLOCK_W=block_w,
            num_warps=4,
        )
        return out
    gate = gate_up[:, :width].float()
    up = gate_up[:, width:].float()
    activated = (float(beta) * torch.tanh(gate / float(beta)) * torch.sigmoid(gate)) * (
        float(linear_beta) * torch.tanh(up / float(linear_beta))
    )
    out.copy_(activated.to(gate_up.dtype))
    return out


@triton.jit
def _sigmoid_gate_kernel(
    value_ptr,
    gate_ptr,
    out_ptr,
    n_elements,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    idx = pid * BLOCK + tl.arange(0, BLOCK)
    mask = idx < n_elements
    v = tl.load(value_ptr + idx, mask=mask, other=0.0).to(tl.float32)
    g = tl.load(gate_ptr + idx, mask=mask, other=0.0).to(tl.float32)
    s = tl.sigmoid(g).to(tl.bfloat16).to(tl.float32)
    tl.store(out_ptr + idx, (v * s).to(tl.bfloat16), mask=mask)


def sigmoid_gate(
    value: torch.Tensor,
    gate: torch.Tensor,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Multiply `value` by `sigmoid(gate)` with BF16 rounding matching PyTorch BF16."""
    if value.shape != gate.shape or value.dtype != gate.dtype or value.device != gate.device:
        raise ValueError("sigmoid_gate requires matching shape, dtype, and device")
    if out is None:
        out = torch.empty_like(value)
    elif out.shape != value.shape or out.dtype != value.dtype or out.device != value.device:
        raise ValueError("sigmoid_gate output must match input shape, dtype, and device")
    n = value.numel()
    if n == 0:
        return out
    if (
        value.is_cuda
        and value.dtype == torch.bfloat16
        and value.is_contiguous()
        and gate.is_contiguous()
        and out.is_contiguous()
    ):
        block = 1024
        grid = (triton.cdiv(n, block),)
        torch.library.wrap_triton(_sigmoid_gate_kernel)[grid](
            value,
            gate,
            out,
            n,
            BLOCK=block,
            num_warps=4,
        )
        return out
    out.copy_(value * torch.sigmoid(gate.float()).to(value.dtype))
    return out


class PreparedSiTUMLP:
    """Dense or shared-expert SiTU MLP, including rank-local intermediate partitions.

    Gate/up projection rows are gate first, then up (`[w1; w3]`). The output is
    a partial hidden vector for sharded weights; the caller owns its collective
    reduction. Mirrors `PreparedMLP`'s projection dispatch and functional API.
    """

    def __init__(
        self,
        source,
        weights: MLPWeights,
        gate_up,
        intermediate,
        output,
        *,
        session,
        name,
        workspace,
        beta: float = 4.0,
        linear_beta: float = 25.0,
        activation_mode="auto",
        gate_up_bias=None,
        down_bias=None,
    ):
        self.gate_up, self.intermediate, self.output = gate_up, intermediate, output
        self.beta, self.linear_beta = float(beta), float(linear_beta)
        if source.ndim != 2 or intermediate.ndim != 2:
            raise ValueError("SiTU MLP input and intermediate must be matrices")
        tokens, hidden = source.shape
        width = intermediate.shape[1]
        for tensor, shape in (
            (source, (tokens, hidden)),
            (gate_up, (tokens, 2 * width)),
            (intermediate, (tokens, width)),
            (output, (tokens, hidden)),
        ):
            if (
                tensor.dtype != torch.bfloat16
                or tensor.shape != shape
                or tensor.device != source.device
                or not tensor.is_contiguous()
            ):
                raise ValueError(
                    "SiTU MLP buffers require contiguous BF16 storage with matching geometry and device"
                )

        def projection(inp, weight, out, suffix, bias):
            from hyperflare_recipes.components.block_fp8 import BlockFP8Projection, BlockFP8Weight

            if isinstance(weight, BlockFP8Weight):
                if bias is not None:
                    raise ValueError("block-FP8 SiTU MLP projections do not support bias")
                return BlockFP8Projection(inp, weight, out, session=session, name=f"{name}.{suffix}")
            if isinstance(weight, torch.Tensor):
                return UnquantizedProjection(
                    inp, weight, out, session=session, name=f"{name}.{suffix}", bias=bias
                )
            return QuantizedProjection(
                inp,
                weight,
                out,
                session=session,
                name=f"{name}.{suffix}",
                workspace=workspace,
                activation_mode=activation_mode,
                bias=bias,
            )

        self.up_projection = projection(source, weights.gate_up, gate_up, "gate_up", gate_up_bias)
        self.down_projection = projection(intermediate, weights.down, output, "down", down_bias)
        self._functional_layout = None

    def functional(self, *, session):
        """Run the prepared SiTU MLP with compiler-local intermediate buffers."""
        if self._functional_layout is None:
            self._functional_layout = (
                tuple(self.intermediate.shape),
                self.intermediate.stride(),
                self.intermediate.dtype,
            )
            self.up_projection.functional(session=session)
            self.down_projection.functional(session=session)
            self.gate_up = self.intermediate = self.output = None
        return self

    def __call__(self, source):
        if self._functional_layout is not None:
            shape, stride, dtype = self._functional_layout
            gate_up = self.up_projection(source)
            intermediate = torch.empty_strided(shape, stride, dtype=dtype, device=source.device)
            situ(gate_up, intermediate, beta=self.beta, linear_beta=self.linear_beta)
            return self.down_projection(intermediate)
        self.up_projection(source)
        situ(self.gate_up, self.intermediate, beta=self.beta, linear_beta=self.linear_beta)
        return self.down_projection(self.intermediate)
