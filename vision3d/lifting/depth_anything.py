from __future__ import annotations

from pathlib import Path


def depth_anything_v2_config(model_path: str):
    from transformers import DepthAnythingConfig

    name = Path(model_path).name.lower()
    if "vits" in name:
        encoder = {
            "hidden_size": 384,
            "num_hidden_layers": 12,
            "num_attention_heads": 6,
            "intermediate_layers": [2, 5, 8, 11],
            "neck_hidden_sizes": [48, 96, 192, 384],
            "fusion_hidden_size": 64,
        }
    elif "vitl" in name:
        encoder = {
            "hidden_size": 1024,
            "num_hidden_layers": 24,
            "num_attention_heads": 16,
            "intermediate_layers": [4, 11, 17, 23],
            "neck_hidden_sizes": [256, 512, 1024, 1024],
            "fusion_hidden_size": 256,
        }
    elif "vitg" in name:
        encoder = {
            "hidden_size": 1536,
            "num_hidden_layers": 40,
            "num_attention_heads": 24,
            "intermediate_layers": [9, 19, 29, 39],
            "neck_hidden_sizes": [1536, 1536, 1536, 1536],
            "fusion_hidden_size": 384,
        }
    else:
        encoder = {
            "hidden_size": 768,
            "num_hidden_layers": 12,
            "num_attention_heads": 12,
            "intermediate_layers": [2, 5, 8, 11],
            "neck_hidden_sizes": [96, 192, 384, 768],
            "fusion_hidden_size": 128,
        }

    out_indices = [idx + 1 for idx in encoder["intermediate_layers"]]
    out_features = [f"stage{idx}" for idx in out_indices]

    return DepthAnythingConfig(
        backbone_config={
            "model_type": "dinov2",
            "image_size": 518,
            "patch_size": 14,
            "num_channels": 3,
            "hidden_size": encoder["hidden_size"],
            "num_hidden_layers": encoder["num_hidden_layers"],
            "num_attention_heads": encoder["num_attention_heads"],
            "mlp_ratio": 4,
            "qkv_bias": True,
            "hidden_act": "gelu",
            "use_swiglu_ffn": False,
            "out_features": out_features,
            "out_indices": out_indices,
            "apply_layernorm": True,
            "reshape_hidden_states": False,
            "use_mask_token": True,
            "layerscale_value": 1.0,
            "layer_norm_eps": 1e-6,
        },
        reassemble_hidden_size=encoder["hidden_size"],
        neck_hidden_sizes=encoder["neck_hidden_sizes"],
        fusion_hidden_size=encoder["fusion_hidden_size"],
        head_hidden_size=32,
        depth_estimation_type="relative",
    )


def convert_depth_anything_v2_state_dict(state_dict: dict) -> dict:
    converted = {}

    def add(target: str, source: str):
        converted[target] = state_dict[source]

    add("backbone.embeddings.cls_token", "pretrained.cls_token")
    add("backbone.embeddings.mask_token", "pretrained.mask_token")
    add("backbone.embeddings.position_embeddings", "pretrained.pos_embed")
    add(
        "backbone.embeddings.patch_embeddings.projection.weight",
        "pretrained.patch_embed.proj.weight",
    )
    add(
        "backbone.embeddings.patch_embeddings.projection.bias",
        "pretrained.patch_embed.proj.bias",
    )

    layer_count = max(
        int(key.split(".")[2])
        for key in state_dict
        if key.startswith("pretrained.blocks.") and key.endswith(".norm1.weight")
    ) + 1

    for idx in range(layer_count):
        original = f"pretrained.blocks.{idx}"
        target = f"backbone.encoder.layer.{idx}"

        add(f"{target}.norm1.weight", f"{original}.norm1.weight")
        add(f"{target}.norm1.bias", f"{original}.norm1.bias")

        query_weight, key_weight, value_weight = state_dict[
            f"{original}.attn.qkv.weight"
        ].chunk(3, dim=0)
        query_bias, key_bias, value_bias = state_dict[f"{original}.attn.qkv.bias"].chunk(
            3, dim=0
        )

        converted[f"{target}.attention.attention.query.weight"] = query_weight
        converted[f"{target}.attention.attention.key.weight"] = key_weight
        converted[f"{target}.attention.attention.value.weight"] = value_weight
        converted[f"{target}.attention.attention.query.bias"] = query_bias
        converted[f"{target}.attention.attention.key.bias"] = key_bias
        converted[f"{target}.attention.attention.value.bias"] = value_bias

        add(f"{target}.attention.output.dense.weight", f"{original}.attn.proj.weight")
        add(f"{target}.attention.output.dense.bias", f"{original}.attn.proj.bias")
        add(f"{target}.layer_scale1.lambda1", f"{original}.ls1.gamma")
        add(f"{target}.norm2.weight", f"{original}.norm2.weight")
        add(f"{target}.norm2.bias", f"{original}.norm2.bias")
        add(f"{target}.mlp.fc1.weight", f"{original}.mlp.fc1.weight")
        add(f"{target}.mlp.fc1.bias", f"{original}.mlp.fc1.bias")
        add(f"{target}.mlp.fc2.weight", f"{original}.mlp.fc2.weight")
        add(f"{target}.mlp.fc2.bias", f"{original}.mlp.fc2.bias")
        add(f"{target}.layer_scale2.lambda1", f"{original}.ls2.gamma")

    add("backbone.layernorm.weight", "pretrained.norm.weight")
    add("backbone.layernorm.bias", "pretrained.norm.bias")

    for idx in range(4):
        add(
            f"neck.reassemble_stage.layers.{idx}.projection.weight",
            f"depth_head.projects.{idx}.weight",
        )
        add(
            f"neck.reassemble_stage.layers.{idx}.projection.bias",
            f"depth_head.projects.{idx}.bias",
        )
        add(f"neck.convs.{idx}.weight", f"depth_head.scratch.layer{idx + 1}_rn.weight")

    for idx in (0, 1, 3):
        add(
            f"neck.reassemble_stage.layers.{idx}.resize.weight",
            f"depth_head.resize_layers.{idx}.weight",
        )
        add(
            f"neck.reassemble_stage.layers.{idx}.resize.bias",
            f"depth_head.resize_layers.{idx}.bias",
        )

    for target_idx, original_idx in enumerate((4, 3, 2, 1)):
        target = f"neck.fusion_stage.layers.{target_idx}"
        original = f"depth_head.scratch.refinenet{original_idx}"

        add(f"{target}.projection.weight", f"{original}.out_conv.weight")
        add(f"{target}.projection.bias", f"{original}.out_conv.bias")

        for target_layer, original_layer in (
            ("residual_layer1", "resConfUnit1"),
            ("residual_layer2", "resConfUnit2"),
        ):
            add(
                f"{target}.{target_layer}.convolution1.weight",
                f"{original}.{original_layer}.conv1.weight",
            )
            add(
                f"{target}.{target_layer}.convolution1.bias",
                f"{original}.{original_layer}.conv1.bias",
            )
            add(
                f"{target}.{target_layer}.convolution2.weight",
                f"{original}.{original_layer}.conv2.weight",
            )
            add(
                f"{target}.{target_layer}.convolution2.bias",
                f"{original}.{original_layer}.conv2.bias",
            )

    add("head.conv1.weight", "depth_head.scratch.output_conv1.weight")
    add("head.conv1.bias", "depth_head.scratch.output_conv1.bias")
    add("head.conv2.weight", "depth_head.scratch.output_conv2.0.weight")
    add("head.conv2.bias", "depth_head.scratch.output_conv2.0.bias")
    add("head.conv3.weight", "depth_head.scratch.output_conv2.2.weight")
    add("head.conv3.bias", "depth_head.scratch.output_conv2.2.bias")

    return converted
