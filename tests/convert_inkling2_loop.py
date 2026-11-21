#!/usr/bin/env python3
"""Convert thinkingmachines/Inkling-2-Mini (model_type: inkling2_loop, NVFP4 modelopt)
into a 4-bit MLX-Serve affine pack.

Usage:
  uv run --python 3.12 --with mlx --with safetensors --with numpy python \
    tests/convert_inkling2_loop.py \
    --src ~/models/Inkling-2-Mini \
    --dst ~/models/Inkling-2-Mini-MLX-Serve-4bit
"""

import argparse
import json
import os
import re
import shutil
from pathlib import Path

import mlx.core as mx

_NVFP4_E2M1_MAX = 6.0
_NVFP4_E4M3_MAX = 448.0

_QUANTIZE_LINEAR_SUFFIXES = (
    ".attn.wq_du.weight",
    ".attn.wk_dv.weight",
    ".attn.wv_dv.weight",
    ".attn.wr_du.weight",
    ".attn.wo_ud.weight",
    ".attn.gate_attn.weight",
    "model.llm.embed.weight",
    "model.llm.unembed.weight",
)

_COPY_FILES = (
    "mtp.safetensors",
    "chat_template.jinja",
    "generation_config.json",
    "processor_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
)

_README_TEMPLATE = """\
---
base_model: thinkingmachines/Inkling-2-Mini
base_model_relation: quantized
library_name: mlx-serve
license: apache-2.0
pipeline_tag: text-generation
tags:
- mlx
- mlx-serve
- inkling2_loop
- moe
- looped-transformer
---

# Inkling-2-Mini-MLX-Serve-4bit

4-bit MLX-Serve affine quantization (`bits=4`, `group_size=64`) of [thinkingmachines/Inkling-2-Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini) (`model_type: inkling2_loop`, 26.4B total / 3.4B–5.4B active looped MoE).

## Conversion details
- **Routed experts (`w13_weight`, `w2_weight`)**: dequantized from source `modelopt` NVFP4 (`group_size=16`, `global_scale = amax * 6 * 448 / outer_scale_denominator`), split into `switch_mlp.gate_proj`, `switch_mlp.up_proj`, and `switch_mlp.down_proj`, and quantized to 4-bit MLX affine (`group_size=64`).
- **Attention & embed/unembed (`wq_du`, `wk_dv`, `wv_dv`, `wr_du`, `wo_ud`, `gate_attn`, `embed`, `unembed`)**: quantized to 4-bit MLX affine (`group_size=64`).
- **Routers, controllers, attention residuals, short-conv weights, norms, `attn_sink_proj`, `rel_logits_proj`**: kept in full/bfloat16 precision (`sconv` transposed `[C, 1, K]` -> `[C, K, 1]`).
"""


def _should_quantize_linear(key: str, arr: mx.array, group_size: int) -> bool:
    if arr.ndim != 2:
        return False
    if arr.shape[0] < 32 or arr.shape[1] % group_size != 0:
        return False
    return key.endswith(_QUANTIZE_LINEAR_SUFFIXES)


def convert(
    src_dir: Path,
    dst_dir: Path,
    bits: int = 4,
    nonexpert_bits: int = 4,
    group_size: int = 64,
    max_shard_bytes: int = 4_000_000_000,
    include_vision: bool = True,
) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    src_weights_path = src_dir / "model-00001-of-00001.safetensors"
    if not src_weights_path.exists():
        raise FileNotFoundError(f"Missing source weights: {src_weights_path}")

    print(f"[convert] Loading {src_weights_path} ...")
    weights = mx.load(str(src_weights_path))

    expert_re = re.compile(r"^(.+\.mlp\.experts)\.(w13|w2)_weight(\w*)$")
    expert_parts: dict[str, dict[str, mx.array]] = {}
    out_tensors: dict[str, mx.array] = {}
    vision_tensors: dict[str, mx.array] = {}

    for k, v in weights.items():
        if k.startswith("model.visual."):
            if include_vision:
                vision_tensors[k] = v
            continue

        m = expert_re.match(k)
        if m:
            prefix, which, suffix = m.groups()
            slot = expert_parts.setdefault(f"{prefix}.{which}", {})
            slot[suffix] = v
            continue

        if k.endswith("_sconv.weight") and v.ndim == 3 and v.shape[1] == 1:
            v = v.swapaxes(1, 2)

        if _should_quantize_linear(k, v, group_size):
            stem = k[: -len(".weight")]
            qw, qs, qb = mx.quantize(
                v.astype(mx.bfloat16), group_size=group_size, bits=nonexpert_bits, mode="affine"
            )
            out_tensors[f"{stem}.weight"] = qw
            out_tensors[f"{stem}.scales"] = qs
            out_tensors[f"{stem}.biases"] = qb
        else:
            out_tensors[k] = v

    print(f"[convert] Dequantizing NVFP4 experts & re-quantizing to {bits}-bit affine (gs={group_size}) ...")
    for key_prefix in sorted(expert_parts.keys()):
        parts = expert_parts[key_prefix]
        w = parts[""]
        scales = parts["_scale"]
        if "_amax" in parts:
            amax = parts["_amax"].astype(mx.float32)
            denom = parts["_outer_scale_denominator"].astype(mx.float32)
            global_scale = amax * (_NVFP4_E2M1_MAX * _NVFP4_E4M3_MAX) / denom
            w_u32 = w.view(mx.uint32)
            w_bf16 = mx.concatenate(
                [
                    mx.dequantize(
                        w_u32[e : e + 1],
                        scales[e : e + 1],
                        group_size=16,
                        bits=4,
                        mode="nvfp4",
                        global_scale=global_scale[e],
                        dtype=mx.bfloat16,
                    )
                    for e in range(w.shape[0])
                ],
                axis=0,
            )
        else:
            # Tail layers (20-23) ship in MXFP8 (8-bit, group_size=32).
            w_bf16 = mx.dequantize(
                w.view(mx.uint32),
                scales,
                group_size=32,
                bits=8,
                mode="mxfp8",
                dtype=mx.bfloat16,
            )

        layer_mlp = key_prefix.rsplit(".experts.", 1)[0]
        which = key_prefix.rsplit(".", 1)[1]
        if which == "w13":
            gate = w_bf16[:, 0::2, :]
            up = w_bf16[:, 1::2, :]
            for proj_name, mat in (("gate_proj", gate), ("up_proj", up)):
                qw, qs, qb = mx.quantize(mat, group_size=group_size, bits=bits, mode="affine")
                mx.eval(qw, qs, qb)
                out_tensors[f"{layer_mlp}.switch_mlp.{proj_name}.weight"] = qw
                out_tensors[f"{layer_mlp}.switch_mlp.{proj_name}.scales"] = qs
                out_tensors[f"{layer_mlp}.switch_mlp.{proj_name}.biases"] = qb
        elif which == "w2":
            qw, qs, qb = mx.quantize(w_bf16, group_size=group_size, bits=bits, mode="affine")
            mx.eval(qw, qs, qb)
            out_tensors[f"{layer_mlp}.switch_mlp.down_proj.weight"] = qw
            out_tensors[f"{layer_mlp}.switch_mlp.down_proj.scales"] = qs
            out_tensors[f"{layer_mlp}.switch_mlp.down_proj.biases"] = qb

    # Shard and save
    shards: list[dict[str, mx.array]] = []
    cur: dict[str, mx.array] = {}
    cur_bytes = 0
    total_bytes = 0
    for k in sorted(out_tensors.keys()):
        arr = out_tensors[k]
        mx.eval(arr)
        nbytes = arr.nbytes
        total_bytes += nbytes
        if cur and cur_bytes + nbytes > max_shard_bytes:
            shards.append(cur)
            cur = {}
            cur_bytes = 0
        cur[k] = arr
        cur_bytes += nbytes
    if cur:
        shards.append(cur)

    weight_map: dict[str, str] = {}
    n_shards = len(shards)
    for idx, shard in enumerate(shards, 1):
        fname = f"model-{idx:05d}-of-{n_shards:05d}.safetensors"
        print(f"[convert] Writing {fname} ({len(shard)} tensors) ...")
        mx.save_safetensors(str(dst_dir / fname), shard)
        for k in shard:
            weight_map[k] = fname

    if vision_tensors:
        vname = "model-vision.safetensors"
        print(f"[convert] Writing {vname} ({len(vision_tensors)} tensors) ...")
        mx.save_safetensors(str(dst_dir / vname), vision_tensors)
        for k, arr in vision_tensors.items():
            total_bytes += arr.nbytes
            weight_map[k] = vname

    index_obj = {
        "metadata": {"total_size": total_bytes},
        "weight_map": weight_map,
    }
    (dst_dir / "model.safetensors.index.json").write_text(json.dumps(index_obj, indent=2) + "\n")

    cfg = json.loads((src_dir / "config.json").read_text())
    cfg["quantization"] = {"group_size": group_size, "bits": bits}
    cfg["quantization_config"] = {"group_size": group_size, "bits": bits, "mode": "affine"}
    if not include_vision:
        cfg.pop("vision_config", None)
    (dst_dir / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")

    for fname in _COPY_FILES:
        src_f = src_dir / fname
        if src_f.exists():
            shutil.copy2(src_f, dst_dir / fname)

    tiktoken_src = src_dir / "tiktoken"
    if tiktoken_src.is_dir():
        shutil.copytree(tiktoken_src, dst_dir / "tiktoken", dirs_exist_ok=True)

    (dst_dir / "README.md").write_text(_README_TEMPLATE)
    print(f"[convert] Done! Total payload: {total_bytes / 1e9:.2f} GB in {dst_dir}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--src", type=Path, required=True)
    p.add_argument("--dst", type=Path, required=True)
    p.add_argument("--bits", type=int, default=4)
    p.add_argument("--nonexpert-bits", type=int, default=4)
    p.add_argument("--group-size", type=int, default=64)
    p.add_argument("--no-vision", action="store_true")
    args = p.parse_args()
    convert(
        args.src,
        args.dst,
        bits=args.bits,
        nonexpert_bits=args.nonexpert_bits,
        group_size=args.group_size,
        include_vision=not args.no_vision,
    )


if __name__ == "__main__":
    main()
