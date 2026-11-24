# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""CPU unit tests for ERNIE-5.5 recipe geometry, operators, contract, and chat template."""

import json
import math
from pathlib import Path

import pytest
import torch

from ignition.preparation import runtime_core_pb2 as p
from ignition.preparation.frontend import PromptEncoder, TemplateEncoding, prepare_frontend

from hyperflare_recipes.components.block_fp8_weights import f32_power_of_two_to_ue8m0
from hyperflare_recipes.components.multimax import MultimaxSelector, apply_multimax
from hyperflare_recipes.components.situ import sigmoid_gate, situ
from hyperflare_recipes.recipes import aliases, recipes
from hyperflare_recipes.recipes.ernie55.attention import (
    DSA_MERGED_WIDTH,
    DSA_TOPK,
    DSA_WINDOW,
    merge_swa_topk_to_physical,
    partial_neox_rope,
)
from hyperflare_recipes.recipes.ernie55.bootstrap import PreparationConfig
from hyperflare_recipes.recipes.ernie55.config import VOCABULARY, ModelConfig
from hyperflare_recipes.recipes.ernie55.contract import FRONTEND
from hyperflare_recipes.recipes.ernie55.moe import ernie55_route_topk, rms_norm_columns
from hyperflare_recipes.recipes.ernie55.pool import continuation_layout, page_bytes
from hyperflare_recipes.recipes.ernie55.recipe import RECIPE


def _sample_config_dict() -> dict:
    ratios = [128] * 43
    for idx in (7, 14, 21, 28, 35, 42):
        ratios[idx] = -2
    return {
        "activation_situ_beta": 4.0,
        "activation_situ_linear_beta": 25.0,
        "architectures": ["Ernie5_5_MoeForCausalLM"],
        "compress_ratios": ratios,
        "compress_rope_theta": 160000.0,
        "dspark_block_size": 7,
        "dspark_markov_rank": 256,
        "dspark_noise_token_id": 1024,
        "dspark_target_layer_ids": [40, 41, 42],
        "first_k_dense_replace": 1,
        "hc_eps": 1e-6,
        "hc_mult": 4,
        "hc_sinkhorn_iters": 20,
        "head_dim": 512,
        "hidden_act": "situ",
        "hidden_size": 4096,
        "hybrid_mla_kv_lora_rank": 512,
        "hybrid_mla_num_attention_heads": 64,
        "hybrid_mla_q_lora_rank": 1024,
        "hybrid_mla_qk_nope_head_dim": 192,
        "hybrid_mla_qk_rope_head_dim": 64,
        "hybrid_mla_v_head_dim": 256,
        "index_head_dim": 128,
        "index_n_heads": 64,
        "index_topk": 2048,
        "intermediate_size": 14336,
        "kv_lora_rank": 512,
        "latent_moe_use_norm": True,
        "max_position_embeddings": 1048576,
        "model_type": "ernie5_5_moe",
        "moe_intermediate_size": 2048,
        "moe_latent_size": 2048,
        "moe_split_feature_routing": True,
        "multimax": "lm_head",
        "n_routed_experts": 512,
        "n_shared_experts": 1,
        "num_attention_heads": 64,
        "num_experts_per_tok": 10,
        "num_hidden_layers": 43,
        "num_key_value_heads": 1,
        "num_nextn_predict_layers": 0,
        "o_groups": 8,
        "o_lora_rank": 1024,
        "q_lora_rank": 1024,
        "qk_nope_head_dim": 448,
        "qk_rope_head_dim": 64,
        "rms_norm_eps": 1e-6,
        "rope_scaling": {
            "beta_fast": 32,
            "beta_slow": 1,
            "factor": 16,
            "original_max_position_embeddings": 65536,
            "type": "yarn",
        },
        "rope_theta": 8000000,
        "scoring_func": "sigmoid",
        "sliding_window": 128,
        "use_gated_attn": True,
        "use_vha_attention": True,
        "v_head_dim": 512,
        "vocab_size": VOCABULARY,
        "quantization_config": {
            "activation_scheme": "dynamic",
            "expert_dtype": "mxfp4",
            "fmt": "e4m3",
            "quant_method": "fp8",
            "scale_fmt": "ue8m0",
            "weight_block_size": [128, 128],
        },
    }


def test_config_and_pool_match_declared_rank_costs():
    cfg = ModelConfig.from_dict(_sample_config_dict())
    assert len(cfg.hca_layers) == 37
    assert cfg.dsa_layers == (7, 14, 21, 28, 35, 42)
    assert cfg.vocabulary == 201216
    assert cfg.draft.layers == 3
    assert cfg.draft.target_layers == (40, 41, 42)
    assert cfg.draft.noise_token == 1024
    assert cfg.draft.block_size == 7

    assert page_bytes(cfg) == RECIPE.ranks[2].page_bytes == 1_274_304
    assert continuation_layout(cfg, dspark=True).row_bytes == RECIPE.ranks[2].state_row_bytes
    assert continuation_layout(cfg, dspark=False).row_bytes == RECIPE.plain_ranks[2].state_row_bytes
    assert "ernie-5.5-flash" in aliases()
    assert any(r.name == "ernie55" for r in recipes())


def test_config_rejects_invalid_variants():
    bad = _sample_config_dict()
    bad["model_type"] = "deepseek_v4"
    with pytest.raises(ValueError, match="Ernie5_5_MoeForCausalLM"):
        ModelConfig.from_dict(bad)

    bad = _sample_config_dict()
    bad["compress_ratios"][7] = 4
    with pytest.raises(ValueError, match="compress_ratios"):
        ModelConfig.from_dict(bad)

    bad = _sample_config_dict()
    bad["quantization_config"]["expert_dtype"] = "fp4"
    with pytest.raises(ValueError, match="MXFP4"):
        ModelConfig.from_dict(bad)


def test_preparation_config_accepts_two_spark_roce_and_rejects_non_dividing_tp():
    roce = PreparationConfig(
        devices=(0, 0),
        hosts=("spark0", "spark1"),
        dspark=True,
        speculative_widths=(1, 2, 3, 4, 5, 6, 7, 8),
        snapshot_budget_bytes=1 << 30,
    )
    assert roce.draft_tokens == 7
    assert roce.use_gpu_packed_speculation is True

    plain = PreparationConfig(devices=(0, 1), dspark=False, snapshot_budget_bytes=1 << 30)
    assert plain.draft_tokens == 0
    assert plain.use_gpu_packed_speculation is False

    with pytest.raises(ValueError, match="2, 4, or 8"):
        PreparationConfig(devices=(0, 1, 2), dspark=False, snapshot_budget_bytes=1 << 30)


def test_frontend_contract_and_prepare_frontend_segmented(tmp_path):
    assert FRONTEND["tools"]["kind"] == p.INVOKE_PARAMETER_XML
    assert FRONTEND["reasoning"]["kind"] == p.ERNIE_THINK
    assert FRONTEND["reasoning"]["open"] == "<ernie:think>"
    assert FRONTEND["reasoning"]["close"] == "</ernie:think>"
    assert FRONTEND["reasoning"]["terminal_markers"] == ["<ernie:tool_call>"]
    assert FRONTEND["assistant_marker"] == "<|ernie:im_start|>assistant\n"
    assert FRONTEND["message_end_markers"] == ["<|ernie:im_end|>"]

    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "bos_token": "<|ERNIE_BEGIN|>",
                "eos_token": "<|ernie:im_end|>",
                "pad_token": "<|ERNIE_PAD|>",
                "clean_up_tokenization_spaces": False,
            }
        )
    )
    fe = prepare_frontend(
        tmp_path,
        eos_token_ids=[2, 8],
        prompt_encoder=PromptEncoder.PROMPT_JINJA,
        template_encoding=TemplateEncoding.TEMPLATE_TOKEN_SEGMENTS,
    )
    assert fe["eos_token_ids"] == [2, 8]
    assert fe["template_encoding"] == int(TemplateEncoding.TEMPLATE_TOKEN_SEGMENTS)
    assert "prompt_encoder" not in fe


def test_f32_power_of_two_to_ue8m0_conversion_and_validation():
    exponents = torch.tensor([-126, -20, -1, 0, 1, 10, 127], dtype=torch.int32)
    scales = torch.pow(2.0, exponents.float())
    ue8m0 = f32_power_of_two_to_ue8m0(scales)
    assert ue8m0.dtype == torch.uint8
    assert torch.equal(ue8m0, (exponents + 127).to(torch.uint8))

    for bad_val in (0.0, -1.0, 1.5, float("inf"), float("-inf"), float("nan"), 2.0**-127):
        with pytest.raises(ValueError, match="exact positive normal power of two"):
            f32_power_of_two_to_ue8m0(torch.tensor([bad_val], dtype=torch.float32))


def test_multimax_matches_reference_and_preserves_neg_inf():
    ranges = torch.tensor([15.625, 15.9375, 8.75, 0.3671875], dtype=torch.float32)
    ts_pos = torch.tensor([0.25, 0.279296875, 0.0625, -0.0301513671875], dtype=torch.float32)
    logits = torch.tensor(
        [
            [float("-inf"), -20.0, 0.0, 8.75, 15.625, 25.0, float("nan")],
            [float("-inf"), 0.3671875, 15.9375, -0.0, 1.0, float("inf"), -100.0],
        ],
        dtype=torch.float32,
    )
    ref = (
        logits
        + ts_pos[0] * (ranges[0] - logits).relu()
        + ts_pos[1] * (logits - ranges[1]).relu()
        + ts_pos[2] * (ranges[2] - logits).relu().square()
        + ts_pos[3] * (logits - ranges[3]).relu().square()
    )
    # Without preserve_neg_inf, positive t[0]/t[2] turn -inf into NaN.
    raw = apply_multimax(logits.clone(), ranges, ts_pos, preserve_neg_inf=False)
    assert torch.isnan(raw[:, 0]).all()

    # With preserve_neg_inf=True (DSpark selector path), -inf stays -inf and finite matches bitwise.
    selector = MultimaxSelector(ranges, ts_pos)
    out = selector.apply(logits.clone(), None, 0, None)
    assert out[:, 0].isneginf().all()
    assert torch.isnan(out[0, 6])
    finite = logits.isfinite()
    assert torch.equal(out[finite].view(torch.int32), ref[finite].view(torch.int32))


def test_situ_and_sigmoid_gate():
    gen = torch.Generator().manual_seed(7)
    gate_up = (torch.randn(4, 32, generator=gen) * 3.0).to(torch.bfloat16)
    got = situ(gate_up)
    g = gate_up[:, :16].float()
    u = gate_up[:, 16:].float()
    want = ((4.0 * torch.tanh(g / 4.0) * torch.sigmoid(g)) * (25.0 * torch.tanh(u / 25.0))).to(
        torch.bfloat16
    )
    assert torch.equal(got, want)

    v = torch.randn(4, 16, generator=gen).to(torch.bfloat16)
    gate = torch.randn(4, 16, generator=gen).to(torch.bfloat16)
    assert torch.equal(sigmoid_gate(v, gate), v * torch.sigmoid(gate.float()).to(torch.bfloat16))


def test_moe_split_routing_and_rms_norm_columns():
    gen = torch.Generator().manual_seed(11)
    logits = torch.randn(5, 1024, generator=gen, dtype=torch.float32) * 2.0
    bias = torch.randn(512, generator=gen, dtype=torch.float32) * 0.1
    scale = torch.rand(512, generator=gen, dtype=torch.float32) + 2.5
    positions = torch.tensor([0, 5, 10, -1, -1], dtype=torch.int64)
    routes = ernie55_route_topk(logits, bias, scale, top_k=10, positions=positions)
    assert routes.ids.shape == (5, 10)
    assert (routes.ids[3:] == -1).all()
    assert (routes.weights[3:] == 0.0).all()

    scores = torch.sigmoid(logits[:3, :512]) + torch.sigmoid(logits[:3, 512:])
    ref_ids = (scores + bias).topk(10, dim=-1, sorted=False).indices
    sel = scores.gather(1, ref_ids)
    ref_w = sel / sel.sum(dim=-1, keepdim=True) * scale[ref_ids]
    assert torch.equal(routes.ids[:3], ref_ids.to(torch.int32))
    torch.testing.assert_close(routes.weights[:3], ref_w)

    x = torch.randn(3, 2048, generator=gen).to(torch.bfloat16)
    w = torch.randn(2048, generator=gen).to(torch.bfloat16)
    full = (x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6) * w.float()).to(
        torch.bfloat16
    )
    left = rms_norm_columns(x, w, col_start=0, col_width=1024, eps=1e-6)
    right = rms_norm_columns(x, w, col_start=1024, col_width=1024, eps=1e-6)
    assert torch.equal(torch.cat((left, right), dim=-1), full)


def test_partial_neox_rope_and_merge_swa_topk_physical():
    gen = torch.Generator().manual_seed(19)
    freqs = 1.0 / (8000000.0 ** (torch.arange(32, dtype=torch.float32) / 32.0))
    q = torch.randn(3, 4, 128, generator=gen).to(torch.bfloat16)
    tail_before = q[..., 64:].clone()
    pos = torch.tensor([0, 17, 1023], dtype=torch.int64)
    partial_neox_rope(q, pos, freqs)
    # Trailing 64 channels must remain untouched.
    assert torch.equal(q[..., 64:], tail_before)

    # Test SWA + topk merge across padding, <128, ==128, middle, and >2176 positions.
    positions = torch.tensor([-1, 10, 127, 300, 5000], dtype=torch.int64)
    # Physical pages are reversed to verify page-table indirection.
    page_table = torch.arange(31, -1, -1, dtype=torch.int32)[None, :].expand(5, -1).contiguous()
    topk_logical = torch.arange(DSA_TOPK, dtype=torch.int32)[None, :].expand(5, -1).contiguous()
    out_phys = torch.empty((5, DSA_MERGED_WIDTH), dtype=torch.int32)
    out_len = torch.empty((5,), dtype=torch.int32)
    merge_swa_topk_to_physical(topk_logical, page_table, positions, out_phys, out_len)

    assert out_len.tolist() == [0, 11, 128, 301, DSA_MERGED_WIDTH]
    assert (out_phys[0] == -1).all()
    # Row 1 (pos=10, seq_len=11): pure SWA tokens 0..10 on page 0 -> physical page 31.
    assert out_phys[1, :11].tolist() == [31 * 256 + i for i in range(11)]
    assert (out_phys[1, 11:] == -1).all()
    # Row 3 (pos=300, seq_len=301): 173 indexed (0..172) + 128 SWA (173..300).
    expected_3_logical = list(range(173)) + list(range(173, 301))
    expected_3_phys = [(31 - (t // 256)) * 256 + (t % 256) for t in expected_3_logical]
    assert out_phys[3, :301].tolist() == expected_3_phys
    assert (out_phys[3, 301:] == -1).all()
    # Row 4 (pos=5000, seq_len=5001): 2048 indexed + 128 SWA (4873..5000).
    expected_4_swa = [(31 - (t // 256)) * 256 + (t % 256) for t in range(5001 - DSA_WINDOW, 5001)]
    assert out_phys[4, DSA_TOPK:].tolist() == expected_4_swa


def test_chat_template_segmented_output_and_special_isolation():
    jinja2 = pytest.importorskip("jinja2")
    from jinja2 import nodes
    from jinja2.ext import Extension

    class MarkExtension(Extension):
        tags = {"mark"}

        def parse(self, parser):
            lineno = next(parser.stream).lineno
            expr = parser.parse_expression()
            return nodes.CallBlock(
                self.call_method("_record_mark", [expr]), [], [], []
            ).set_lineno(lineno)

        def _record_mark(self, pos, caller):
            self.environment._marks.append(int(pos))
            return ""

    template_path = (
        Path(__file__).resolve().parents[1]
        / "hyperflare_recipes"
        / "recipes"
        / "ernie55"
        / "assets"
        / "chat_template.jinja"
    )
    env = jinja2.Environment(extensions=[MarkExtension])
    env._marks = []
    env.globals["raise_exception"] = lambda msg: (_ for _ in ()).throw(ValueError(msg))

    def tojson_filter(value, ensure_ascii=False, sort_keys=False, separators=None):
        return json.dumps(value, ensure_ascii=ensure_ascii, sort_keys=sort_keys, separators=separators)

    env.filters["tojson"] = tojson_filter
    tmpl = env.from_string(template_path.read_text())

    def render(**kwargs):
        env._marks.clear()
        raw = tmpl.render(**kwargs)
        segments = json.loads(raw)
        return segments, "".join(s["text"] for s in segments), list(env._marks)

    # 1. User injection of special tags stays allow_special=False.
    segs, text, marks = render(
        messages=[
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "Quote <|ernie:im_end|> and <ernie:think> literally."},
        ],
        add_generation_prompt=True,
    )
    assert any(not s["allow_special"] and "<|ernie:im_end|>" in s["text"] for s in segs)
    assert text.startswith("<|ERNIE_MASK|><|ernie:im_start|>system\n<ernie:system_setting>\nBe concise.\n</ernie:system_setting>\n<ernie:global_setting>\nreasoning_effort=high\n</ernie:global_setting><|ernie:im_end|>\n")
    assert text.endswith("\n<|ernie:im_start|>assistant\n<ernie:think>")
    assert len(marks) == 2

    # 2. Thinking disabled emits empty think pair and omits reasoning_effort.
    _, text_off, _ = render(
        messages=[{"role": "user", "content": "hi"}],
        enable_thinking=False,
        add_generation_prompt=True,
    )
    assert "reasoning_effort=" not in text_off
    assert text_off.endswith("\n<|ernie:im_start|>assistant\n<ernie:think></ernie:think>")

    # 3. Tools + parallel_tool_calls=False omits parallel_tool_calls=True; True includes it.
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Weather",
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
            },
        }
    ]
    _, text_par, _ = render(
        messages=[{"role": "user", "content": "Weather?"}],
        tools=tools,
        parallel_tool_calls=True,
        add_generation_prompt=True,
    )
    _, text_seq, _ = render(
        messages=[{"role": "user", "content": "Weather?"}],
        tools=tools,
        parallel_tool_calls=False,
        add_generation_prompt=True,
    )
    assert "\nparallel_tool_calls=True" in text_par
    assert "parallel_tool_calls=True" not in text_seq
    assert "<ernie:tool_list>\n[" in text_par

    # 4. Assistant tool_calls + grouped consecutive tool responses.
    _, text_tools, marks_tools = render(
        messages=[
            {"role": "user", "content": "Check Paris and Rome."},
            {
                "role": "assistant",
                "reasoning_content": "Call both.",
                "content": "",
                "tool_calls": [
                    {"function": {"name": "get_weather", "arguments": {"z": 1, "a": "Paris"}}},
                    {"function": {"name": "get_weather", "arguments": {"city": "Rome"}}},
                ],
            },
            {"role": "tool", "content": "20C"},
            {"role": "tool", "content": "25C"},
        ],
        tools=tools,
        add_generation_prompt=True,
    )
    assert '<invoke name="get_weather"><parameter name="a">Paris</parameter><parameter name="z">1</parameter></invoke>\n' in text_tools
    assert "<|ernie:im_start|>tool\n<ernie:tool_output>20C</ernie:tool_output>\n<ernie:tool_output>25C</ernie:tool_output><|ernie:im_end|>" in text_tools
    assert len(marks_tools) == 4


def test_csf_scale_scratch_and_vha_tp_mix_and_sparse_mla_import():
    from hyperflare_recipes.recipes.ernie55.attention import (
        PreparedSparseMLA,
        PreparedVHAPostMix,
    )
    from hyperflare_recipes.recipes.ernie55.weights import Ernie55WeightLoader

    assert PreparedSparseMLA is not None

    loader = Ernie55WeightLoader.__new__(Ernie55WeightLoader)
    loader.device = torch.device("cpu")
    loader.scale_scratch = None
    loader.expert_scale_compression = None
    assert loader._expert_scale_scratch(512, 2048, 1024) is None

    loader.expert_scale_compression = "csf"
    first = loader._expert_scale_scratch(512, 2048, 1024)
    second = loader._expert_scale_scratch(512, 2048, 1024)
    assert first is second
    assert isinstance(first, tuple) and len(first) == 2
    assert first[0].shape == (512, 2048, 64) and first[0].dtype == torch.uint8 and first[0].is_contiguous()
    assert first[1].shape == (512, 2048, 32) and first[1].dtype == torch.uint8 and first[1].is_contiguous()
    assert first[0].untyped_storage().data_ptr() != first[1].untyped_storage().data_ptr()

    # Verify TP=2 VHA FP32 -> BF16 pair gather -> FP32 sum matches full-head reference.
    gen = torch.Generator().manual_seed(29)
    rows, heads, rank, head_dim = 4, 64, 16, 64
    u_full = torch.randn(heads, rank, generator=gen, dtype=torch.bfloat16)
    v_full = torch.randn(heads, rank, generator=gen, dtype=torch.bfloat16)
    o_full = torch.randn(rows, heads, head_dim, generator=gen, dtype=torch.bfloat16)
    z_ref = torch.bmm(u_full.float().t().expand(rows, -1, -1), o_full.float())
    out_ref = (o_full.float() + torch.bmm(v_full.float().expand(rows, -1, -1), z_ref)).to(torch.bfloat16)

    half = heads // 2
    mix0 = PreparedVHAPostMix(
        u_full[:half], v_full[:half], tokens=rows, head_dim=head_dim, world=1,
        gather_runtime=None, coordinator=None, session=None, name="vha0",
    )
    mix1 = PreparedVHAPostMix(
        u_full[half:], v_full[half:], tokens=rows, head_dim=head_dim, world=1,
        gather_runtime=None, coordinator=None, session=None, name="vha1",
    )
    mix0.world = mix1.world = 2

    def make_gather(rank_id, peer_tensor):
        def gather(local_bf16):
            return torch.cat((local_bf16, peer_tensor), dim=-1) if rank_id == 0 else torch.cat((peer_tensor, local_bf16), dim=-1)
        return gather

    z0 = torch.bmm(u_full[:half].float().t().expand(rows, -1, -1), o_full[:, :half].float())
    z1 = torch.bmm(u_full[half:].float().t().expand(rows, -1, -1), o_full[:, half:].float())
    z0_bf16 = z0.contiguous().view(torch.bfloat16).view(rows, 2 * rank * head_dim)
    z1_bf16 = z1.contiguous().view(torch.bfloat16).view(rows, 2 * rank * head_dim)

    out0 = mix0._mix_chunk(o_full[:, :half], make_gather(0, z1_bf16))
    out1 = mix1._mix_chunk(o_full[:, half:], make_gather(1, z0_bf16))
    torch.testing.assert_close(torch.cat((out0, out1), dim=1), out_ref, rtol=1e-2, atol=1e-2)


def test_dsa_kv_a_576_padded_block_fp8_loading():
    from hyperflare_recipes.recipes.ernie55.weights import Ernie55WeightLoader

    loader = Ernie55WeightLoader.__new__(Ernie55WeightLoader)
    loader.device = torch.device("cpu")
    loader.config = ModelConfig.from_dict(_sample_config_dict())
    loader._session = type("_S", (), {"flush": lambda self: None})()
    loader._require_open = lambda: None
    loader._copy = lambda dst, src: dst.copy_(src)
    w = torch.randn(576, 4096).to(torch.float8_e4m3fn)
    # 5 scale rows for ceil(576 / 128) = 5, 32 cols for 4096 / 128.
    s = torch.full((5, 32), 1.0, dtype=torch.float32)
    loader._sources = {
        "m.kv_a_proj_with_mqa.weight": w,
        "m.kv_a_proj_with_mqa.weight_scale_inv": s,
    }
    vals, scales = loader._dense_blocks("m.kv_a_proj_with_mqa", rows=(0, 640), pad=True)
    assert vals.shape == (640, 4096)
    assert scales.shape == (5, 32)
    assert torch.equal(vals[:576], w)
    assert (vals[576:].float() == 0).all()
    assert (scales == 127).all()
