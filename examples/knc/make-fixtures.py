#!/usr/bin/env python3
"""Small deterministic architecture fixtures, not trained models or speed benchmarks."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys
import zlib

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "gguf-py"))
import gguf
import numpy as np


def png(path, color):
    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
    raw = b"".join(b"\0" + bytes(color) * 32 for _ in range(32))
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 32, 32, 8, 2, 0, 0, 0))
                     + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


def text_model(output, arch):
    w = gguf.GGUFWriter(output, arch)
    w.add_name("KNC random architecture fixture - NOT a trained model")
    tokens = ["<unk>", "<s>", "</s>"] + [f"<0x{i:02X}>" for i in range(256)]
    types = [2, 3, 3] + [6] * 256
    tokens += ["\u2581"] + list("abcdefghijklmnopqrstuvwxyz")
    types += [1] * 27
    specials = ["<im_start>", "<im_end>", "<patch_start>", "<patch_end>", "<patch_newline>"]
    tokens += specials
    types += [4] * len(specials)
    w.add_tokenizer_model("llama")
    w.add_token_list(tokens)
    w.add_token_scores([0.0] * len(tokens))
    w.add_token_types(types)
    w.add_bos_token_id(1)
    w.add_eos_token_id(2)
    w.add_unk_token_id(0)
    w.add_add_bos_token(False)
    w.add_vocab_size(len(tokens))
    w.add_context_length(512)
    w.add_embedding_length(256)
    w.add_block_count(3)
    w.add_feed_forward_length(256)
    w.add_head_count(8)
    w.add_head_count_kv([0, 2, 0] if arch in ("lfm2", "lfm2moe") else 2)
    w.add_layer_norm_rms_eps(1e-5)
    if arch in ("lfm2", "lfm2moe"):
        w.add_shortconv_l_cache(3)
    if arch == "lfm2moe":
        w.add_leading_dense_block_count(1)
        w.add_expert_count(4)
        w.add_expert_used_count(2)
        w.add_expert_feed_forward_length(256)
        w.add_expert_gating_func(gguf.ExpertGatingFuncType.SIGMOID)
        w.add_expert_weights_norm(True)
    rng = np.random.default_rng(104729)

    def tensor(name, shape, norm=False):
        data = np.ones(shape) if norm else rng.standard_normal(shape) * 0.02
        w.add_tensor(name, data.astype(np.float32))

    tensor("token_embd.weight", (len(tokens), 256))
    tensor("token_embd_norm.weight" if arch in ("lfm2", "lfm2moe") else "output_norm.weight", (256,), True)
    tensor("output.weight", (len(tokens), 256))
    for i in range(3):
        p = f"blk.{i}."
        for name in ("attn_norm", "ffn_norm"):
            tensor(p + name + ".weight", (256,), True)
        if arch in ("lfm2", "lfm2moe") and i != 1:
            tensor(p + "shortconv.conv.weight", (256, 3))
            tensor(p + "shortconv.in_proj.weight", (768, 256))
            tensor(p + "shortconv.out_proj.weight", (256, 256))
        else:
            for name, rows in (("attn_q", 256), ("attn_k", 64), ("attn_v", 64), ("attn_output", 256)):
                tensor(p + name + ".weight", (rows, 256))
            if arch != "llama":
                for name in ("attn_q_norm", "attn_k_norm"):
                    tensor(p + name + ".weight", (32,), True)
        if arch == "lfm2moe" and i > 0:
            tensor(p + "ffn_gate_inp.weight", (4, 256))
            tensor(p + "exp_probs_b.bias", (4,))
            for name in ("ffn_gate_exps", "ffn_down_exps", "ffn_up_exps"):
                tensor(p + name + ".weight", (4, 256, 256))
        else:
            for name in ("ffn_gate", "ffn_up", "ffn_down"):
                tensor(p + name + ".weight", (256, 256))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


def projector(output):
    w = gguf.GGUFWriter(output, "clip")
    w.add_name("KNC tiny Step3-VL projector fixture - NOT trained")
    w.add_string("clip.projector_type", "step3vl")
    w.add_bool("clip.has_vision_encoder", True)
    for key, value in {"embedding_length": 32, "feed_forward_length": 64, "block_count": 1,
                       "projection_dim": 256, "attention.head_count": 4, "image_size": 56,
                       "patch_size": 14, "preproc_image_size": 56}.items():
        w.add_uint32("clip.vision." + key, value)
    w.add_float32("clip.vision.attention.layer_norm_epsilon", 1e-5)
    w.add_array("clip.vision.image_mean", [0.5, 0.5, 0.5])
    w.add_array("clip.vision.image_std", [0.5, 0.5, 0.5])
    rng = np.random.default_rng(130363)

    def tensor(name, shape, norm=False):
        data = np.ones(shape) if norm else rng.standard_normal(shape) * 0.02
        # Match the converter: learned positions must remain F32 for CPU addition.
        dtype = np.float32 if len(shape) == 1 or name == "v.position_embd.weight" else np.float16
        w.add_tensor(name, data.astype(dtype))

    tensor("v.patch_embd.weight", (32, 3, 14, 14))
    tensor("v.position_embd.weight", (16, 32))
    for name in ("attn_q", "attn_k", "attn_v", "attn_out"):
        tensor("v.blk.0." + name + ".weight", (32, 32))
    for name in ("ln1", "ln2"):
        tensor("v.blk.0." + name + ".weight", (32,), True)
    tensor("v.blk.0.ffn_up.weight", (64, 32))
    tensor("v.blk.0.ffn_down.weight", (32, 64))
    tensor("mm.0.weight", (64, 32, 3, 3))
    tensor("mm.1.weight", (128, 64, 3, 3))
    tensor("mm.model.fc.weight", (256, 128))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--quantizer", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    for arch in ("llama", "lfm2", "lfm2moe", "qwen3"):
        base = args.output / f"{arch}-f32.gguf"
        text_model(base, arch)
        for quant in ("Q3_K_M", "Q4_K_M", "Q5_K_M"):
            dest = args.output / f"{arch}-{quant}.gguf"
            with dest.with_suffix(".log").open("w") as log:
                subprocess.run([str(args.quantizer), str(base), str(dest), quant], check=True, stdout=log, stderr=subprocess.STDOUT)
            reader = gguf.GGUFReader(dest)
            counts = Counter(t.tensor_type.name for t in reader.tensors)
            assert counts[quant.split("_M")[0]], counts
    projector(args.output / "step3vl-mmproj.gguf")
    png(args.output / "red.png", (255, 0, 0))
    png(args.output / "blue.png", (0, 0, 255))
    (args.output / "text.txt").write_text("hello world", encoding="utf-8")
    (args.output / "image.txt").write_text("<__media__> describe", encoding="utf-8")
    manifest = {"trained": False, "hardware_acceptance": False, "seed_text": 104729, "seed_vision": 130363,
                "files": {p.name: {"sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "bytes": p.stat().st_size}
                          for p in args.output.iterdir() if p.suffix != ".log"}}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
