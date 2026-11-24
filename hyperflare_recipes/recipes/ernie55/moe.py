# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""ERNIE-5.5 split-feature top-10 routing and Latent MoE (4096 -> 2048 -> 512 MXFP4 SiTU -> 4096)."""

from dataclasses import dataclass

import torch
import triton
import triton.language as tl

from ignition.components.collectives import PreparedAllReduce
from hyperflare_recipes.components.linear import QuantizedProjection, UnquantizedProjection
from hyperflare_recipes.components.mlp import MLPScratch
from hyperflare_recipes.components.moe import PreparedRoutedExperts, RouterScratch
from hyperflare_recipes.components.situ import PreparedSiTUMLP


@triton.jit
def _ernie55_split_sigmoid_kernel(
    logits_ptr,
    scores_ptr,
    rows,
    row_stride,
    NUM_EXPERTS: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    if row >= rows:
        return
    col = tl.arange(0, NUM_EXPERTS)
    base = row * row_stride + col
    l0 = tl.load(logits_ptr + base).to(tl.float32)
    l1 = tl.load(logits_ptr + base + NUM_EXPERTS).to(tl.float32)
    s = tl.sigmoid(l0) + tl.sigmoid(l1)
    tl.store(scores_ptr + row * NUM_EXPERTS + col, s)


@triton.jit
def _rms_norm_columns_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    rows,
    x_stride,
    out_stride,
    eps,
    DIM: tl.constexpr,
    COL_START: tl.constexpr,
    COL_WIDTH: tl.constexpr,
):
    """Full-row RMSNorm(2048) writing only the local TP slice `[COL_START : COL_START + COL_WIDTH]`."""
    row = tl.program_id(0).to(tl.int64)
    if row >= rows:
        return
    full = tl.arange(0, DIM)
    x_full = tl.load(x_ptr + row * x_stride + full).to(tl.float32)
    inv_rms = tl.rsqrt(tl.sum(x_full * x_full, axis=0) * (1.0 / DIM) + eps)
    cols = tl.arange(0, COL_WIDTH)
    src_cols = COL_START + cols
    x_slice = tl.load(x_ptr + row * x_stride + src_cols).to(tl.float32)
    w_slice = tl.load(w_ptr + src_cols).to(tl.float32)
    tl.store(out_ptr + row * out_stride + cols, (x_slice * inv_rms * w_slice).to(tl.bfloat16))


def rms_norm_columns(
    x: torch.Tensor,
    weight: torch.Tensor,
    *,
    col_start: int,
    col_width: int,
    eps: float = 1e-6,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute RMSNorm across the full row of `x` and emit only `[col_start : col_start + col_width]`."""
    rows, dim = x.shape
    if out is None:
        out = torch.empty((rows, col_width), dtype=x.dtype, device=x.device)
    if rows == 0:
        return out
    if x.is_cuda and x.dtype == torch.bfloat16 and dim == 2048 and col_width in (256, 512, 1024, 2048):
        torch.library.wrap_triton(_rms_norm_columns_kernel)[(rows,)](
            x,
            weight,
            out,
            rows,
            x.stride(0),
            out.stride(0),
            float(eps),
            DIM=dim,
            COL_START=col_start,
            COL_WIDTH=col_width,
            num_warps=4,
        )
        return out
    xf = x.float()
    inv_rms = torch.rsqrt(xf.square().mean(dim=-1, keepdim=True) + float(eps))
    sliced = xf[:, col_start : col_start + col_width] * inv_rms * weight[col_start : col_start + col_width].float()
    out.copy_(sliced.to(x.dtype))
    return out


def ernie55_route_topk(
    logits: torch.Tensor,
    bias: torch.Tensor,
    scale: torch.Tensor,
    *,
    top_k: int = 10,
    positions: torch.Tensor | None = None,
) -> RouterScratch:
    """ERNIE-5.5 split-feature routing: sigmoid(l0) + sigmoid(l1) + biased top-k + renorm + per-expert scale.

    `logits` has shape `[rows, 2 * E]` (concatenated outputs of `gate.weight` on
    `x[:, :2048]` and `gate.weight_1` on `x[:, 2048:]`). Inactive rows
    (`positions < 0`) receive expert ID `-1` and weight `0.0`.
    """
    rows, double_experts = logits.shape
    num_experts = double_experts // 2
    if logits.is_cuda and logits.dtype == torch.float32 and num_experts == 512:
        scores = torch.empty((rows, num_experts), dtype=torch.float32, device=logits.device)
        if rows > 0:
            torch.library.wrap_triton(_ernie55_split_sigmoid_kernel)[(rows,)](
                logits, scores, rows, logits.stride(0), NUM_EXPERTS=num_experts, num_warps=4
            )
    else:
        lf = logits.float()
        scores = torch.sigmoid(lf[:, :num_experts]) + torch.sigmoid(lf[:, num_experts:])

    biased = scores + bias.float()
    topk_vals, topk_ids = torch.topk(biased, k=top_k, dim=-1, sorted=False)
    ids = topk_ids.to(torch.int32)
    selected = scores.gather(1, topk_ids)
    denom = selected.sum(dim=-1, keepdim=True)
    weights = torch.where(denom > 0, selected / denom, torch.zeros_like(selected))
    weights = weights * scale.float()[topk_ids]
    if positions is not None:
        live = positions[:, None] >= 0
        ids = torch.where(live, ids, torch.full_like(ids, -1))
        weights = torch.where(live, weights, torch.zeros_like(weights))
    return RouterScratch(topk_vals, ids, weights)


@dataclass(frozen=True)
class LatentMoEScratch:
    router_logits: torch.Tensor
    router: RouterScratch
    latent_in: torch.Tensor
    routed_partial: torch.Tensor
    routed_latent: torch.Tensor
    latent_normed: torch.Tensor
    fc2_out: torch.Tensor
    shared: MLPScratch
    mlp_partial: torch.Tensor
    mlp_output: torch.Tensor

    @classmethod
    def allocate(cls, tokens: int, world: int, device, *, experts: int = 512, top_k: int = 10):
        def zeros(*shape, dtype=torch.bfloat16):
            return torch.zeros(shape, dtype=dtype, device=device)

        latent = 2048
        return cls(
            router_logits=zeros(tokens, 2 * experts, dtype=torch.float32),
            router=RouterScratch.allocate(tokens, device, experts=experts, top_k=top_k),
            latent_in=zeros(tokens, latent),
            routed_partial=zeros(tokens, latent),
            routed_latent=zeros(tokens, latent),
            latent_normed=zeros(tokens, latent // world),
            fc2_out=zeros(tokens, 4096),
            shared=MLPScratch.allocate(tokens, 4096, 2048 // world, device),
            mlp_partial=zeros(tokens, 4096),
            mlp_output=zeros(tokens, 4096),
        )

    def rows(self, tokens: int):
        return LatentMoEScratch(
            router_logits=self.router_logits[:tokens],
            router=self.router.rows(tokens),
            latent_in=self.latent_in[:tokens],
            routed_partial=self.routed_partial[:tokens],
            routed_latent=self.routed_latent[:tokens],
            latent_normed=self.latent_normed[:tokens],
            fc2_out=self.fc2_out[:tokens],
            shared=self.shared.rows(tokens),
            mlp_partial=self.mlp_partial[:tokens],
            mlp_output=self.mlp_output[:tokens],
        )


class PreparedLatentMoE:
    """ERNIE-5.5 Latent MoE: split-feature router + fc1_latent -> 512 MXFP4 SiTU -> AR(2048) -> norm -> fc2_latent + shared SiTU -> AR(4096)."""

    def __init__(
        self,
        weights,
        source: torch.Tensor,
        scratch: LatentMoEScratch,
        *,
        shard,
        eps: float,
        beta: float,
        linear_beta: float,
        top_k: int,
        latent_runtime,
        hidden_runtime,
        coordinator,
        session,
        name: str,
        workspace,
    ):
        self.weights = weights
        self.eps = float(eps)
        self.top_k = int(top_k)
        self.latent_start, self.latent_end = shard.partition(2048, alignment=128)
        self.latent_width = self.latent_end - self.latent_start

        # Split-feature router projections on x[:, :2048] and x[:, 2048:] -> concatenated [T, 1024] FP32.
        num_experts = weights.router.shape[0] // 2
        self.router_w0 = weights.router[:num_experts]
        self.router_w1 = weights.router[num_experts:]
        self.router_proj0 = UnquantizedProjection(
            source[:, :2048].contiguous(),
            self.router_w0,
            scratch.router_logits[:, :num_experts].contiguous(),
            session=session,
            name=name + ".router0",
        ).functional(session=session)
        self.router_proj1 = UnquantizedProjection(
            source[:, 2048:].contiguous(),
            self.router_w1,
            scratch.router_logits[:, num_experts:].contiguous(),
            session=session,
            name=name + ".router1",
        ).functional(session=session)

        self.fc1_latent = QuantizedProjection(
            source,
            weights.fc1_latent,
            scratch.latent_in,
            session=session,
            name=name + ".fc1_latent",
            workspace=workspace,
            activation_mode="auto",
        ).functional(session=session)

        self.experts = PreparedRoutedExperts(
            scratch.latent_in,
            weights.experts,
            scratch.router.ids,
            scratch.router.weights,
            scratch.routed_partial,
            session=session,
            name=name + ".experts",
            workspace=workspace,
        ).functional(session=session)

        self.latent_reduce = PreparedAllReduce(
            scratch.routed_partial,
            scratch.routed_latent,
            runtime=latent_runtime,
            coordinator=coordinator,
            session=session,
            name=name + ".latent_reduce",
        ).functional(session=session)

        self.fc2_latent = QuantizedProjection(
            scratch.latent_normed,
            weights.fc2_latent,
            scratch.fc2_out,
            session=session,
            name=name + ".fc2_latent",
            workspace=workspace,
            activation_mode="auto",
        ).functional(session=session)

        self.shared = PreparedSiTUMLP(
            source,
            weights.shared,
            scratch.shared.gate_up,
            scratch.shared.intermediate,
            scratch.shared.output,
            session=session,
            name=name + ".shared",
            workspace=workspace,
            beta=beta,
            linear_beta=linear_beta,
            activation_mode="auto",
        ).functional(session=session)

        self.hidden_reduce = PreparedAllReduce(
            scratch.mlp_partial,
            scratch.mlp_output,
            runtime=hidden_runtime,
            coordinator=coordinator,
            session=session,
            name=name + ".hidden_reduce",
        ).functional(session=session)

    def __call__(self, source: torch.Tensor, metadata) -> torch.Tensor:
        l0 = self.router_proj0(source[:, :2048].contiguous())
        l1 = self.router_proj1(source[:, 2048:].contiguous())
        logits = torch.cat((l0, l1), dim=-1)
        routes = ernie55_route_topk(
            logits,
            self.weights.correction,
            self.weights.routed_scaling_factor,
            top_k=self.top_k,
            positions=metadata.positions,
        )
        latent_in = self.fc1_latent(source)
        routed_partial = self.experts(latent_in, routes.ids, routes.weights)
        routed_latent = self.latent_reduce(routed_partial)
        normed_slice = rms_norm_columns(
            routed_latent,
            self.weights.latent_norm,
            col_start=self.latent_start,
            col_width=self.latent_width,
            eps=self.eps,
        )
        routed_out = self.fc2_latent(normed_slice)
        shared_out = self.shared(source)
        return self.hidden_reduce(routed_out + shared_out)
