# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Block-128 FP8, native MXFP4 SiTU experts, optional FlashIndexer, and bundled DSpark loader."""

from dataclasses import dataclass, replace
from pathlib import Path

import torch

from hyperflare_recipes.components.block_fp8_weights import BlockFP8WeightLoader
from hyperflare_recipes.components.weights import MLPWeights
from ignition.preparation.persistent import (
    allocated_tensor_bytes,
    persistent_cat,
    persistent_copy,
    persistent_empty,
)
from ignition.preparation.rebuild_manifest import weight_container

from .config import VOCABULARY, ModelConfig


@weight_container(
    (
        "router",
        "correction",
        "routed_scaling_factor",
        "fc1_latent",
        "latent_norm",
        "fc2_latent",
        "experts",
        "shared",
    )
)
@dataclass(frozen=True)
class LatentMoEWeights:
    router: torch.Tensor
    correction: torch.Tensor
    routed_scaling_factor: torch.Tensor
    fc1_latent: object
    latent_norm: torch.Tensor
    fc2_latent: object
    experts: object
    shared: MLPWeights


@weight_container(("attention", "mlp", "experts", "residual", "is_dsa"))
@dataclass(frozen=True)
class LayerWeights:
    attention: dict
    mlp: MLPWeights | None
    experts: LatentMoEWeights | None
    residual: dict
    is_dsa: bool = False


@weight_container(
    (
        "layers",
        "main_projection",
        "context_kv",
        "main_norm",
        "norm",
        "head_mix",
        "markov_embedding",
        "markov_head",
        "confidence_head",
        "draft_head",
    )
)
@dataclass(frozen=True)
class DSparkWeights:
    layers: tuple[LayerWeights, ...]
    main_projection: object
    context_kv: object
    main_norm: torch.Tensor
    norm: torch.Tensor
    head_mix: dict
    markov_embedding: torch.Tensor
    markov_head: object
    confidence_head: torch.Tensor | None = None
    draft_head: object | None = None

    @classmethod
    def load(cls, loader, *, markov_head_quantization=None):
        c, shard = loader.config.draft, loader.shard
        mtp = "model.mtp"
        layers = tuple(loader.load_layer(i, draft=True) for i in range(c.layers))
        main_projection = loader.load_linear(f"{mtp}.feature_projection")
        context_kv = loader.load_linear_group(
            tuple(f"{mtp}.{i}.self_attn.linear_kv_proj" for i in range(c.layers))
        )
        main_norm = loader.load_tensor(f"{mtp}.feature_norm.weight")
        norm = loader.load_tensor(f"{mtp}.norm.weight")
        head_mix = loader.load_head_mix(f"{mtp}.mhc_contract.")
        # Checkpoint stores markov_w1 as [markov_rank, vocab] (256, 201216); transpose to [vocab, 256].
        w1 = loader.load_tensor(f"{mtp}.markov_head.markov_w1.weight", temporary=True)
        markov_embedding = persistent_copy(w1.transpose(0, 1).contiguous())
        markov_head = loader.load_tensor(
            f"{mtp}.markov_head.markov_w2.weight",
            extent=slice(*shard.partition(VOCABULARY, alignment=64)),
            temporary=markov_head_quantization == "nvfp4",
        )
        conf_key = f"{mtp}.confidence_head.projection.weight"
        confidence_head = (
            loader.load_tensor(conf_key) if conf_key in loader._sources else None
        )
        return cls(
            layers=layers,
            main_projection=main_projection,
            context_kv=context_kv,
            main_norm=main_norm,
            norm=norm,
            head_mix=head_mix,
            markov_embedding=markov_embedding,
            markov_head=markov_head,
            confidence_head=confidence_head,
        )


@dataclass(frozen=True)
class ModelWeights:
    config: ModelConfig
    shard: object
    embedding: torch.Tensor
    norm: torch.Tensor
    head: torch.Tensor
    multimax_ranges: torch.Tensor
    multimax_ts: torch.Tensor
    head_mix: dict
    layers: tuple[LayerWeights, ...]
    draft: DSparkWeights | None
    allocated_bytes: int
    scale_scratch: tuple[torch.Tensor, ...] | None = None

    @classmethod
    def load(
        cls,
        manifest,
        device,
        *,
        shard,
        progress=None,
        dspark=True,
        draft_head_quantization=None,
        markov_head_quantization=None,
        force_moe_decode_a16=False,
        expert_scale_compression="csf",
        enable_flashindexer=True,
    ):
        from hyperflare_recipes.components.vocabulary import nvfp4_draft_head

        config = ModelConfig.load(manifest.metadata_path)
        before = allocated_tensor_bytes(device)
        with Ernie55WeightLoader(
            manifest,
            device,
            shard=shard,
            progress=progress,
            dspark=dspark,
            force_moe_decode_a16=force_moe_decode_a16,
            expert_scale_compression=expert_scale_compression,
            enable_flashindexer=enable_flashindexer,
        ) as loader:
            embedding, norm = loader.load_tensors(("model.embed_tokens.weight", "model.norm.weight"))
            head = loader.load_tensor(
                "lm_head.weight", extent=slice(*shard.partition(VOCABULARY, alignment=64))
            )
            multimax_ranges = loader.load_tensor("model.lm_head.multimax_ranges", dtype=torch.float32)
            multimax_ts = loader.load_tensor("model.lm_head.multimax_ts", dtype=torch.float32)
            head_mix = loader.load_head_mix("model.mhc_contract.")
            layers = tuple(loader.load_layer(i) for i in range(config.layers))
            draft = (
                DSparkWeights.load(loader, markov_head_quantization=markov_head_quantization)
                if dspark
                else None
            )
            if draft is not None:
                if draft_head_quantization == "nvfp4":
                    draft = replace(draft, draft_head=nvfp4_draft_head(head))
                if markov_head_quantization == "nvfp4":
                    draft = replace(draft, markov_head=nvfp4_draft_head(draft.markov_head))
        return cls(
            config=config,
            shard=shard,
            embedding=embedding,
            norm=norm,
            head=head,
            multimax_ranges=multimax_ranges,
            multimax_ts=multimax_ts,
            head_mix=head_mix,
            layers=layers,
            draft=draft,
            allocated_bytes=allocated_tensor_bytes(device) - before,
            scale_scratch=loader.scale_scratch,
        )


def _load_flashindexer_tensors(snapshot: Path, device) -> dict[str, torch.Tensor]:
    """Load optional `flashindexer/flashindexer.safetensors` gates if present in snapshot."""
    path = Path(snapshot) / "flashindexer" / "flashindexer.safetensors"
    if not path.is_file():
        return {}
    from safetensors.torch import load_file

    raw = load_file(str(path), device="cpu")
    result = {}
    for key, tensor in raw.items():
        if key.endswith("flashindexer_bias"):
            result[key] = persistent_copy(tensor.to(device=device, dtype=torch.float32))
        else:
            result[key] = persistent_copy(tensor.to(device=device, dtype=torch.bfloat16))
    return result


class Ernie55WeightLoader(BlockFP8WeightLoader):
    scale_suffix = ".weight_scale_inv"

    def __init__(
        self,
        manifest,
        device,
        *,
        shard,
        progress=None,
        prefixes=None,
        dspark=True,
        force_moe_decode_a16=False,
        expert_scale_compression="csf",
        enable_flashindexer=True,
    ):
        super().__init__(manifest, device, progress=progress, prefixes=prefixes)
        self.config = ModelConfig.load(manifest.metadata_path)
        self.shard = shard
        self.dspark = dspark
        self.force_moe_decode_a16 = bool(force_moe_decode_a16)
        self.expert_scale_compression = expert_scale_compression
        self.enable_flashindexer = bool(enable_flashindexer)
        self.scale_scratch = None
        self._flashindexer = (
            _load_flashindexer_tensors(manifest.metadata_path, device)
            if self.enable_flashindexer
            else {}
        )

    def _expert_scale_scratch(self, count: int, hidden: int, local: int):
        if self.expert_scale_compression is None:
            return None
        if self.scale_scratch is None:
            def buffer(rows, columns):
                return torch.empty(
                    (count, (rows + 255) // 256 * 256, (columns + 7) // 8 * 8),
                    dtype=torch.uint8,
                    device=self.device,
                )

            self.scale_scratch = (
                buffer(2 * local, hidden // 32),
                buffer(hidden, local // 32),
            )
        return self.scale_scratch

    def _selected_names(self):
        return {
            name
            for name in super()._selected_names()
            if self.dspark or not name.startswith("model.mtp.")
        }

    def load_head_mix(self, prefix: str) -> dict:
        # Checkpoint stores hc_head_fn as [4 * hidden, 4]; transpose to [4, 4 * hidden] FP32.
        fn = self.load_tensor(prefix + "hc_head_fn", dtype=torch.float32, temporary=True)
        if fn.shape[0] != 4:
            fn = fn.transpose(0, 1).contiguous()
        return {
            "fn": persistent_copy(fn),
            "base": self.load_tensor(prefix + "hc_head_base", dtype=torch.float32),
            "scale": self.load_tensor(prefix + "hc_head_scale", dtype=torch.float32),
        }

    def _load_mhc_block(self, prefix: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Checkpoint stores mapping_proj.weight as [16384, 24]; PreparedMHC expects [24, 16384] FP32.
        fn = self.load_tensor(prefix + ".mapping_proj.weight", dtype=torch.float32, temporary=True)
        if fn.shape[0] != 24:
            fn = fn.transpose(0, 1).contiguous()
        base = self.load_tensor(prefix + ".bias", dtype=torch.float32)
        alpha_pre = self.load_tensor(prefix + ".alpha_pre", dtype=torch.float32, temporary=True).reshape(1)
        alpha_post = self.load_tensor(prefix + ".alpha_post", dtype=torch.float32, temporary=True).reshape(1)
        alpha_res = self.load_tensor(prefix + ".alpha_res", dtype=torch.float32, temporary=True).reshape(1)
        scale = persistent_cat((alpha_pre, alpha_post, alpha_res), dim=0)
        return persistent_copy(fn), scale, base

    def load_compressor(self, prefix: str) -> dict:
        return {
            "values": self.load_tensor(prefix + ".linear_wkv.weight", temporary=True),
            "gates": self.load_tensor(prefix + ".linear_wgate.weight", temporary=True),
            "norm": self.load_tensor(prefix + ".norm.weight"),
            "ape": self.load_tensor(prefix + ".ape"),
        }

    def _load_vha(self, attn_prefix: str) -> dict:
        head_slice = slice(*self.shard.partition(64))
        return {
            "vha_u": self.load_tensor(attn_prefix + ".vha_postmix_U", extent=head_slice),
            "vha_v": self.load_tensor(attn_prefix + ".vha_postmix_V", extent=head_slice),
        }

    def _load_hca_attention(self, layer: int, *, draft: bool = False) -> dict:
        prefix = f"model.mtp.{layer}" if draft else f"model.layers.{layer}"
        attn = prefix + ".self_attn"
        text, shard = self.config.text, self.shard
        heads = shard.partition(64 * 512, alignment=512)
        groups = shard.partition(text["o_groups"] * text["o_lora_rank"], alignment=text["o_lora_rank"])
        local_groups = (groups[1] - groups[0]) // text["o_lora_rank"]
        group_projs = tuple(
            self._pack_blocks(
                [
                    self._dense_blocks(
                        attn + ".linear_o_group_proj",
                        rows=(
                            groups[0] + g * text["o_lora_rank"],
                            groups[0] + (g + 1) * text["o_lora_rank"],
                        ),
                        temporary=True,
                        weight_name=attn + ".linear_o_group_proj",
                        scale_name=attn + ".linear_o_group_proj_scale_inv",
                    )
                ],
                dim=0,
            )
            for g in range(local_groups)
        )
        attention = {
            "qkv_a": self.load_linear_group((attn + ".linear_q_down_proj", attn + ".linear_kv_proj")),
            "q_b": self.load_linear(attn + ".linear_q_up_proj", rows=heads),
            "o_group": group_projs,
            "o_gate": self.load_linear(attn + ".gate_proj", rows=groups),
            "o_proj": self.load_linear(attn + ".o_proj", columns=groups),
            "q_norm": self.load_tensor(attn + ".q_layernorm.weight"),
            "kv_norm": self.load_tensor(attn + ".kv_layernorm.weight"),
            "sink": self.load_tensor(
                attn + ".core_attention.attn_sink",
                extent=slice(*shard.partition(64)),
                dtype=torch.float32,
            ),
            **self._load_vha(attn),
        }
        if not draft:
            compressor = self.load_compressor(attn + ".core_attention.compressor")
            parts = (compressor.pop("values"), compressor.pop("gates"))
            attention["compressor"] = compressor
            attention["compression"] = persistent_cat(parts)
        return attention

    def _load_dsa_attention(self, layer: int) -> dict:
        attn = f"model.layers.{layer}.self_attn"
        text, shard = self.config.text, self.shard
        num_heads = text["hybrid_mla_num_attention_heads"]
        qk_nope = text["hybrid_mla_qk_nope_head_dim"]
        qk_rope = text["hybrid_mla_qk_rope_head_dim"]
        qk_dim = qk_nope + qk_rope
        v_dim = text["hybrid_mla_v_head_dim"]
        kv_lora = text["hybrid_mla_kv_lora_rank"]

        head_range = shard.partition(num_heads)
        local_heads = head_range[1] - head_range[0]
        q_rows = (head_range[0] * qk_dim, head_range[1] * qk_dim)
        v_rows = (head_range[0] * v_dim, head_range[1] * v_dim)
        kv_b_rows = (head_range[0] * (qk_nope + v_dim), head_range[1] * (qk_nope + v_dim))

        # Dequantize local shard of kv_b_proj into absorbed W_uk ([H_local, 192, 512]) and W_uv ([H_local, 512, 256]).
        kv_b = self.load_dequantized_linear(attn + ".kv_b_proj", rows=kv_b_rows, dtype=torch.bfloat16)
        kv_b = kv_b.view(local_heads, qk_nope + v_dim, kv_lora)
        w_uk = persistent_copy(kv_b[:, :qk_nope, :].contiguous())
        w_uv = persistent_copy(kv_b[:, qk_nope:, :].transpose(1, 2).contiguous())
        del kv_b

        idx = attn + ".core_attention.indexer"
        # Pack wk ([128, 4096]) and weights_proj ([64, 4096]) for a single BF16 projection.
        wk = self.load_tensor(idx + ".wk.weight", temporary=True)
        wproj = self.load_tensor(idx + ".weights_proj.weight", temporary=True)
        index_kw = persistent_cat((wk, wproj), dim=0)

        kv_a_padded = ((kv_lora + qk_rope + self.block_size - 1) // self.block_size) * self.block_size
        attention = {
            "qkv_a": self.load_linear_group(
                (attn + ".q_a_proj", attn + ".kv_a_proj_with_mqa"),
                row_ranges=(None, (0, kv_a_padded)),
                pad=True,
            ),
            "q_b": self.load_linear(attn + ".q_b_proj", rows=q_rows),
            "q_norm": self.load_tensor(attn + ".q_a_layernorm.weight"),
            "kv_norm": self.load_tensor(attn + ".kv_a_layernorm.weight"),
            "w_uk": w_uk,
            "w_uv": w_uv,
            "o_gate": self.load_linear(attn + ".gate_proj", rows=v_rows),
            "o_proj": self.load_linear(attn + ".o_proj", columns=v_rows),
            "sink": self.load_tensor(
                attn + ".core_attention.softmax_offset",
                extent=slice(*head_range),
                dtype=torch.float32,
            ),
            "index_q": self.load_linear(idx + ".wq_b"),
            "index_kw": index_kw,
            "index_k_norm_weight": self.load_tensor(idx + ".k_norm.weight", dtype=torch.float32),
            "index_k_norm_bias": self.load_tensor(idx + ".k_norm.bias", dtype=torch.float32),
            **self._load_vha(attn),
        }

        # Attach optional FlashIndexer prefill head-selection gate if bundled in snapshot.
        gate_candidates = (
            f"{idx}.linear_gate.weight",
            f"model.layers.{layer}.attn.indexer.linear_gate.weight",
        )
        bias_candidates = (
            f"{idx}.flashindexer_bias",
            f"model.layers.{layer}.attn.indexer.flashindexer_bias",
        )
        gate_w = next((self._flashindexer[k] for k in gate_candidates if k in self._flashindexer), None)
        gate_b = next((self._flashindexer[k] for k in bias_candidates if k in self._flashindexer), None)
        if gate_w is not None and gate_b is not None:
            attention["flashindexer_gate"] = gate_w
            attention["flashindexer_bias"] = gate_b
        return attention

    def _load_dense_mlp(self, prefix: str) -> MLPWeights:
        mlp = prefix + ".mlp"
        interval = self.shard.partition(self.config.text["intermediate_size"], alignment=128)
        return MLPWeights(
            self.load_linear_group((mlp + ".gate_proj", mlp + ".up_proj"), row_ranges=(interval, interval)),
            self.load_linear(mlp + ".down_proj", columns=interval),
        )

    def _load_latent_moe(self, prefix: str) -> LatentMoEWeights:
        moe = prefix + ".block_sparse_moe"
        text, shard = self.config.text, self.shard
        latent_size = text["moe_latent_size"]
        inter_interval = shard.partition(text["moe_intermediate_size"], alignment=128)
        latent_interval = shard.partition(latent_size, alignment=128)
        shared_interval = shard.partition(text["moe_intermediate_size"], alignment=128)

        # Fused split-feature router: stack gate.weight ([512, 4096]) and gate.weight_1 ([512, 4096]) in FP32.
        w0 = self.load_tensor(moe + ".gate.weight", dtype=torch.float32, temporary=True)
        w1 = self.load_tensor(moe + ".gate.weight_1", dtype=torch.float32, temporary=True)
        router = persistent_cat((w0, w1), dim=0)
        correction = self.load_tensor(moe + ".e_score_correction_bias", dtype=torch.float32)
        routed_scaling = self.load_tensor(moe + ".gate.routed_scaling_factor_param", dtype=torch.float32)

        fc1_latent = self.load_linear(moe + ".fc1_latent_proj")
        latent_norm = self.load_tensor(moe + ".latent_norm.weight")
        fc2_latent = self.load_linear(moe + ".fc2_latent_proj", columns=latent_interval)

        activation_mode = "a16" if self.force_moe_decode_a16 else "a8"
        local_inter = inter_interval[1] - inter_interval[0]
        scale_scratch = self._expert_scale_scratch(
            text["n_routed_experts"], latent_size, local_inter
        )
        experts = self.load_fp4_experts(
            [f"{moe}.experts.{i}" for i in range(text["n_routed_experts"])],
            hidden_size=latent_size,
            intermediate_size=text["moe_intermediate_size"],
            format="mxfp4",
            activation_mode=activation_mode,
            intermediate_range=inter_interval,
            scale_compression=self.expert_scale_compression,
            scale_scratch=scale_scratch,
            projection_names=("w1", "w3", "w2"),
            scale_name="weight_scale_inv",
            activation="situ",
            w13_layout="w31",
        )
        shared = MLPWeights(
            self.load_linear_group(
                (moe + ".shared_experts.w1", moe + ".shared_experts.w3"),
                row_ranges=(shared_interval, shared_interval),
            ),
            self.load_linear(moe + ".shared_experts.w2", columns=shared_interval),
        )
        return LatentMoEWeights(
            router=router,
            correction=correction,
            routed_scaling_factor=routed_scaling,
            fc1_latent=fc1_latent,
            latent_norm=latent_norm,
            fc2_latent=fc2_latent,
            experts=experts,
            shared=shared,
        )

    def load_layer(self, layer: int, *, draft: bool = False) -> LayerWeights:
        prefix = f"model.mtp.{layer}" if draft else f"model.layers.{layer}"
        is_dsa = (not draft) and self.config.is_dsa_layer(layer)
        attention = self._load_dsa_attention(layer) if is_dsa else self._load_hca_attention(layer, draft=draft)
        if not draft and layer < self.config.text["first_k_dense_replace"]:
            mlp = self._load_dense_mlp(prefix)
            experts = None
        else:
            mlp = None
            experts = self._load_latent_moe(prefix)

        attn_fn, attn_scale, attn_base = self._load_mhc_block(prefix + ".self_attention_hyper_connection")
        ffn_fn, ffn_scale, ffn_base = self._load_mhc_block(prefix + ".mlp_hyper_connection")
        residual = {
            "hc_attn_fn": attn_fn,
            "hc_attn_scale": attn_scale,
            "hc_attn_base": attn_base,
            "hc_ffn_fn": ffn_fn,
            "hc_ffn_scale": ffn_scale,
            "hc_ffn_base": ffn_base,
            "attn_norm.weight": self.load_tensor(prefix + ".input_layernorm.weight"),
            "ffn_norm.weight": self.load_tensor(prefix + ".post_attention_layernorm.weight"),
        }
        return LayerWeights(
            attention=attention,
            mlp=mlp,
            experts=experts,
            residual=residual,
            is_dsa=is_dsa,
        )
