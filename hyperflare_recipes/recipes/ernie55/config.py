# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Geometry and compression contracts for ERNIE-5.5-Flash-Preview."""

import json
from dataclasses import dataclass
from pathlib import Path

MODEL_ID = "baidu/ERNIE-5.5-Flash-Preview"
VOCABULARY = 201216
DSA_LAYER_RATIO = -2
HCA_LAYER_RATIO = 128


@dataclass(frozen=True)
class Compressor:
    layer: int
    ratio: int
    head_dim: int
    index: bool = False

    @property
    def prefix(self):
        return f"layer{self.layer}.{'index_' if self.index else ''}compressor"

    @property
    def overlap(self):
        return False

    @property
    def state_shape(self):
        return self.ratio, self.head_dim


@dataclass(frozen=True)
class DSparkConfig:
    layers: int
    block_size: int
    noise_token: int
    target_layers: tuple[int, ...]
    markov_rank: int
    experts: int
    top_k: int


@dataclass(frozen=True)
class ModelConfig:
    text: dict
    quantization: dict

    @classmethod
    def load(cls, snapshot):
        return cls.from_dict(json.loads((Path(snapshot) / "config.json").read_text()))

    @classmethod
    def from_dict(cls, config):
        if config.get("model_type") != "ernie5_5_moe" or config.get("architectures") != [
            "Ernie5_5_MoeForCausalLM"
        ]:
            raise ValueError("ERNIE-5.5 requires Ernie5_5_MoeForCausalLM (model_type=ernie5_5_moe)")
        for name, expected in {
            "hidden_size": 4096,
            "num_hidden_layers": 43,
            "num_attention_heads": 64,
            "num_key_value_heads": 1,
            "head_dim": 512,
            "qk_nope_head_dim": 448,
            "qk_rope_head_dim": 64,
            "v_head_dim": 512,
            "q_lora_rank": 1024,
            "kv_lora_rank": 512,
            "o_lora_rank": 1024,
            "o_groups": 8,
            "hybrid_mla_num_attention_heads": 64,
            "hybrid_mla_q_lora_rank": 1024,
            "hybrid_mla_kv_lora_rank": 512,
            "hybrid_mla_qk_nope_head_dim": 192,
            "hybrid_mla_qk_rope_head_dim": 64,
            "hybrid_mla_v_head_dim": 256,
            "index_n_heads": 64,
            "index_head_dim": 128,
            "index_topk": 2048,
            "sliding_window": 128,
            "vocab_size": VOCABULARY,
            "intermediate_size": 14336,
            "first_k_dense_replace": 1,
            "moe_intermediate_size": 2048,
            "moe_latent_size": 2048,
            "n_routed_experts": 512,
            "n_shared_experts": 1,
            "num_experts_per_tok": 10,
            "hidden_act": "situ",
            "scoring_func": "sigmoid",
            "hc_mult": 4,
            "hc_sinkhorn_iters": 20,
            "dspark_markov_rank": 256,
        }.items():
            if config.get(name) != expected:
                raise ValueError(
                    f"ERNIE-5.5 {name} requires {expected!r}, got {config.get(name)!r}"
                )
        if config.get("use_vha_attention") is not True or config.get("use_gated_attn") is not True:
            raise ValueError("ERNIE-5.5 requires VHA post-mix and gated attention")
        if config.get("moe_split_feature_routing") is not True or config.get("latent_moe_use_norm") is not True:
            raise ValueError("ERNIE-5.5 requires split-feature routing and latent MoE normalization")
        if config.get("multimax") != "lm_head":
            raise ValueError("ERNIE-5.5 requires multimax='lm_head'")

        quant = config.get("quantization_config") or {}
        expert_dtype = quant.get("expert_dtype", config.get("expert_dtype"))
        if (
            quant.get("quant_method") != "fp8"
            or quant.get("weight_block_size") != [128, 128]
            or quant.get("scale_fmt") != "ue8m0"
            or expert_dtype != "mxfp4"
            or quant.get("activation_scheme") != "dynamic"
        ):
            raise ValueError(
                "ERNIE-5.5 requires block-128 UE8M0 FP8 dense weights and MXFP4 routed experts"
            )

        result = cls(config, quant)
        ratios = config.get("compress_ratios", [])
        if (
            len(ratios) != result.layers
            or any(type(r) is not int or r not in (HCA_LAYER_RATIO, DSA_LAYER_RATIO) for r in ratios)
        ):
            raise ValueError("ERNIE-5.5 requires 43 compress_ratios entries in {128, -2}")

        targets = config.get("dspark_target_layer_ids", [])
        noise = config.get("dspark_noise_token_id", -1)
        block = config.get("dspark_block_size", 0)
        if (
            type(block) is not int
            or block <= 0
            or type(noise) is not int
            or not 0 <= noise < VOCABULARY
            or not targets
            or any(type(i) is not int or not 0 <= i < result.layers for i in targets)
            or targets != sorted(set(targets))
        ):
            raise ValueError("ERNIE-5.5 requires valid bundled DSpark metadata")
        return result

    @property
    def vocabulary(self):
        return self.text["vocab_size"]

    @property
    def layers(self):
        return self.text["num_hidden_layers"]

    @property
    def hca_layers(self):
        return tuple(
            layer
            for layer, ratio in enumerate(self.text["compress_ratios"])
            if ratio == HCA_LAYER_RATIO
        )

    @property
    def dsa_layers(self):
        return tuple(
            layer
            for layer, ratio in enumerate(self.text["compress_ratios"])
            if ratio == DSA_LAYER_RATIO
        )

    def is_dsa_layer(self, layer: int) -> bool:
        return self.text["compress_ratios"][layer] == DSA_LAYER_RATIO

    @property
    def situ_beta(self) -> float:
        return float(self.text.get("activation_situ_beta", 4.0))

    @property
    def situ_linear_beta(self) -> float:
        return float(self.text.get("activation_situ_linear_beta", 25.0))

    @property
    def draft(self):
        c = self.text
        # Bundled DSpark checkpoints always contain 3 SWA-only MTP layers (model.mtp.0..2)
        # even when num_nextn_predict_layers is 0 in config.json.
        draft_layers = len(c["dspark_target_layer_ids"])
        return DSparkConfig(
            layers=draft_layers,
            block_size=c["dspark_block_size"],
            noise_token=c["dspark_noise_token_id"],
            target_layers=tuple(c["dspark_target_layer_ids"]),
            markov_rank=c["dspark_markov_rank"],
            experts=c["n_routed_experts"],
            top_k=c["num_experts_per_tok"],
        )

    @property
    def compressors(self):
        return tuple(
            Compressor(layer, HCA_LAYER_RATIO, self.text["head_dim"])
            for layer in self.hca_layers
        )
