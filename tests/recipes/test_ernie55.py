# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import pytest
import torch

from hyperflare_recipes.recipes import resolve
from hyperflare_recipes.recipes.ernie55.recipe import RECIPE


def test_ernie55_recipe_registration_and_metadata() -> None:
    resolved = resolve("ernie-5.5-flash")
    assert resolved.recipe.name == "ernie55"
    assert resolved.recipe is RECIPE
    assert any(c.architecture == "Ernie5_5_MoeForCausalLM" for c in RECIPE.checkpoints)
    assert RECIPE.served_model_name == "ERNIE-5.5-Flash-Preview"
    assert RECIPE.prefill_chunk == 4096
    assert RECIPE.max_draft_tokens == 7


def test_ernie55_multihost_spark_preparation_config() -> None:
    from hyperflare_recipes.recipes.ernie55.bootstrap import PreparationConfig

    plain = PreparationConfig(
        devices=(0, 0),
        hosts=("spark-ky00", "spark-ky01"),
        max_active=8,
        prefill_chunk_size=4096,
        dspark=False,
    )
    assert plain.draft_tokens == 0
    assert plain.devices == (0, 0)
    assert plain.hosts == ("spark-ky00", "spark-ky01")

    spec = PreparationConfig(
        devices=(0, 0),
        hosts=("spark-ky00", "spark-ky01"),
        max_active=16,
        prefill_chunk_size=4096,
        dspark=True,
        speculative_widths=(1, 2, 4, 6, 8),
        draft_head_quantization="nvfp4",
        markov_head_quantization="nvfp4",
    )
    assert spec.draft_tokens == 7
    assert spec.packed_verification is True
    assert spec.draft_prefix_selection == "gpu"


def test_ernie55_flashindexer_broadcast_for_multihost_join(tmp_path, monkeypatch) -> None:
    from types import SimpleNamespace

    from safetensors.torch import save_file

    from hyperflare_recipes.components.parallel import TensorParallelShard
    from hyperflare_recipes.recipes.ernie55.weights import ErnieWeightLoader

    rank0_dir = tmp_path / "rank0"
    rank1_dir = tmp_path / "rank1"
    (rank0_dir / "flashindexer").mkdir(parents=True)
    rank1_dir.mkdir(parents=True)

    dsa_layers = (6, 13)
    tensors = {}
    for layer in dsa_layers:
        tensors[f"model.layers.{layer}.self_attn.core_attention.indexer.linear_gate.weight"] = (
            torch.arange(64 * 4, dtype=torch.float32).reshape(64, 4).to(torch.bfloat16)
        )
        tensors[f"model.layers.{layer}.self_attn.core_attention.indexer.flashindexer_bias"] = (
            torch.linspace(-1.0, 1.0, 64, dtype=torch.float32)
        )
    save_file(tensors, str(rank0_dir / "flashindexer" / "flashindexer.safetensors"))

    shared_payload: list[bytes | None] = [None]

    def fake_get_ranks(_group):
        return [0, 1]

    def fake_broadcast(obj_list, *, src, group):
        assert src == 0
        assert group == "fake-group"
        if obj_list[0] is not None:
            shared_payload[0] = obj_list[0]
        else:
            obj_list[0] = shared_payload[0]

    monkeypatch.setattr(torch.distributed, "get_process_group_ranks", fake_get_ranks)
    monkeypatch.setattr(torch.distributed, "broadcast_object_list", fake_broadcast)

    def make_loader(rank: int):
        loader = object.__new__(ErnieWeightLoader)
        loader.device = torch.device("cpu")
        loader.shard = TensorParallelShard(rank, 2)
        loader.config = SimpleNamespace(dsa_layers=dsa_layers)
        return loader

    # Rank 0 reads flashindexer.safetensors and broadcasts raw bytes to rank 1,
    # whose synthetic ignition-join snapshot directory has no flashindexer/ subdir.
    r0_gates = make_loader(0)._load_flashindexer_weights(rank0_dir, group="fake-group")
    r1_gates = make_loader(1)._load_flashindexer_weights(rank1_dir, group="fake-group")

    assert set(r0_gates.keys()) == set(dsa_layers)
    assert set(r1_gates.keys()) == set(dsa_layers)
    for layer in dsa_layers:
        torch.testing.assert_close(r1_gates[layer][0], r0_gates[layer][0])
        torch.testing.assert_close(r1_gates[layer][1], r0_gates[layer][1])
        assert r1_gates[layer][0].dtype == torch.bfloat16
        assert r1_gates[layer][1].dtype == torch.float32

    # When absent on rank 0, both ranks receive {}.
    shared_payload[0] = None
    assert make_loader(0)._load_flashindexer_weights(rank1_dir, group="fake-group") == {}
    assert make_loader(1)._load_flashindexer_weights(rank1_dir, group="fake-group") == {}


@pytest.mark.skipif(
    os.environ.get("FLARE_RUN_CUDA_TESTS") != "1" or not torch.cuda.is_available(),
    reason="CUDA opt-in",
)
def test_multimax_inplace_matches_reference() -> None:
    from hyperflare_recipes.components.markov_sampling import apply_multimax_inplace

    device = torch.device("cuda", 0)
    ranges = (-2.0, 2.0, -4.0, 4.0)
    ts = (-0.25, 0.35, -0.05, 0.02)
    x = torch.linspace(-8.0, 8.0, 4096, device=device, dtype=torch.float32).reshape(2, 2048)
    x[0, 0] = float("-inf")

    expected = x.clone()
    r0, r1, r2, r3 = ranges
    t0, t1, t2, t3 = ts
    a = (r0 - expected).clamp(min=0.0)
    b = (expected - r1).clamp(min=0.0)
    c = (r2 - expected).clamp(min=0.0)
    d = (expected - r3).clamp(min=0.0)
    expected = expected + t0 * a + t1 * b + t2 * (c * c) + t3 * (d * d)
    expected[0, 0] = float("-inf")

    actual = x.clone()
    apply_multimax_inplace(actual, (ranges, ts))
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(
    os.environ.get("FLARE_RUN_CUDA_TESTS") != "1" or not torch.cuda.is_available(),
    reason="CUDA opt-in",
)
def test_ernie55_route_matches_reference_and_pads_safely() -> None:
    from hyperflare_recipes.recipes.ernie55.decoder import ernie55_route

    device = torch.device("cuda", 0)
    torch.manual_seed(17)
    T, E, K = 8, 512, 10
    logits = torch.randn((T, 2 * E), device=device, dtype=torch.bfloat16)
    bias = torch.randn((E,), device=device, dtype=torch.float32) * 0.1
    scale = torch.linspace(2.5, 3.5, E, device=device, dtype=torch.float32)
    positions = torch.tensor([0, 1, 2, 3, 4, 5, -1, -1], device=device, dtype=torch.int32)

    ids, weights = ernie55_route(logits, bias, scale, positions, top_k=K, renormalize=True)
    assert weights.shape == (T, K)
    assert ids.shape == (T, K)
    # Padded rows must have ids == -1 and weights == 0.0
    assert (ids[6:] == -1).all()
    assert (weights[6:] == 0.0).all()
    assert ((ids[:6] >= 0) & (ids[:6] < E)).all()

    scores = torch.sigmoid(logits[:6, :E].float()) + torch.sigmoid(logits[:6, E:].float())
    biased = scores + bias.unsqueeze(0)
    ref_ids = torch.topk(biased, k=K, dim=-1, sorted=False).indices
    ref_w = scores.gather(-1, ref_ids)
    ref_w = ref_w / (ref_w.sum(dim=-1, keepdim=True) + 1e-20)
    ref_w = ref_w * scale[ref_ids]

    for row in range(6):
        got_order = torch.argsort(ids[row])
        ref_order = torch.argsort(ref_ids[row])
        assert torch.equal(ids[row][got_order].long(), ref_ids[row][ref_order].long())
        torch.testing.assert_close(
            weights[row][got_order], ref_w[row][ref_order], rtol=1e-4, atol=1e-4
        )
