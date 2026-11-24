# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Prepared graph inputs, hybrid HCA/DSA continuation storage, and ERNIE frontend contract."""

from ignition.preparation import contract as c
from ignition.preparation import runtime_core_pb2 as p

from .config import DSA_LAYER_RATIO, HCA_LAYER_RATIO, VOCABULARY

FRONTEND = c.frontend(
    tools=p.INVOKE_PARAMETER_XML,
    reasoning=p.ERNIE_THINK,
    reasoning_open="<ernie:think>",
    reasoning_close="</ernie:think>",
    terminal_markers=["<ernie:tool_call>"],
    assistant_marker="<|ernie:im_start|>assistant\n",
    message_end_markers=["<|ernie:im_end|>"],
    prefix_boundary=c.prefix_boundary(
        ["system"],
        required_start="<|ERNIE_MASK|><|ernie:im_start|>system\n",
        required_end="<|ernie:im_end|>\n",
    ),
)


def prepare(
    signatures,
    buffers,
    pool,
    *,
    page_tokens,
    draft_lag,
    rows=1,
    embedding_tables=(),
    regions=(),
):
    regions = list(regions)
    ratios = pool.config.text["compress_ratios"]
    for name, tensor in pool.pages.items():
        layer_str, kind = name.split(".", 1)
        layer = int(layer_str.removeprefix("layer"))
        ratio = ratios[layer]
        if ratio == HCA_LAYER_RATIO:
            tokens_per_record = HCA_LAYER_RATIO
        elif ratio == DSA_LAYER_RATIO:
            tokens_per_record = 1
        else:
            raise ValueError(f"unexpected compress ratio {ratio!r} at layer {layer}")
        region_kind = p.PAGED_LATENT if kind == "latent_page" else p.PAGED_LATENT
        regions.append(
            c.history_region(
                buffers,
                "cache." + name,
                tensor,
                region_kind,
                tokens_per_record=tokens_per_record,
            )
        )
    inputs = {kind: c.text_inputs(kind) for kind in {s.kind for s in signatures.values()}}
    continuation = c.state_view("state_pool", pool.layout.row_bytes)
    consumed = pool.layout.region("consumed_tokens")
    cursor = (
        None
        if draft_lag is None
        else dict(
            state=c.state_view(
                "state_pool",
                pool.layout.row_bytes,
                offset=consumed.offset,
                nbytes=consumed.nbytes,
            ),
            lag_tokens=draft_lag,
        )
    )
    pending = c.state_view("request_pending_token", 8)
    graphs = c.graph_contracts(
        signatures,
        inputs,
        continuation=continuation,
        pending=pending,
        cursor=cursor,
        rows=rows,
        embedding_tables=embedding_tables,
    )
    return c.model_contract(
        vocabulary=VOCABULARY,
        input_vocabulary=VOCABULARY,
        page_tokens=page_tokens,
        page_tables=[
            dict(buffer="page_tables", page_offset=0),
            *(
                [dict(buffer="draft_page_tables", page_offset=1)]
                if "draft_page_tables" in buffers
                else []
            ),
        ],
        regions=[
            dict(
                kind=p.CONTINUATION_STATE,
                storage=continuation,
                tokens_per_record=1,
                window_tokens=0,
            ),
            *regions,
        ],
        graphs=graphs,
        frontend=FRONTEND,
    )
