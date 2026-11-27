# VENDORED from llm.asm by Alexey Fateev (alesha-pro), https://github.com/alesha-pro/llm.asm (MIT);
# Qwen3.5 v3 export from our fork github.com/tcsenpai/llm.asm (scripts/export_weights.py). Unmodified below except the
# ringzero changes marked "ringzero": --v3 writes a Qwen3 checkpoint as v3 (Qwen3-Embedding-0.6B, weights without
# the "model." prefix) for the QWEN35 kernel build.
"""
Export Qwen3-0.6B or Qwen3.5-2B (BF16 safetensors + tokenizer.json) into a compact,
64-byte aligned, directly mmap-able .asmbin file for pure x86-64 asm-llm.

Usage:
  Qwen3-0.6B  -> v2 (FP16 scales, default) or v1 (--scale-dtype fp32); the original uniform-layer layout.
  Qwen3.5-2B  -> v3, auto-detected from config.json (needs the kernel built with MODEL_ARCH=qwen35).
  python3 scripts/export_weights.py --model-dir models/qwen3-0.6b-bf16 --out models/qwen3-0.6b-q8_0.asmbin
  python3 scripts/export_weights.py --model-dir models/qwen3.5-2b-bf16 --out models/qwen35-2b-q8_0-v3.asmbin

Format v3 (Qwen3.5 only, see docs/qwen35-2b/FORMAT_V3.md): 256-byte header + per-layer descriptor table (layer type +
16 absolute u64 tensor offsets per layer, see shared/constants.inc). The kernel finds every layer tensor
through the table; layers no longer need a uniform stride. v1/v2: 128-byte header, uniform layer stride.

Weight format: Per-row split Q8_0 (block size = 32, symmetric int8 in [-127, 127]):
  Each row of K elements (K multiple of 32) is stored as:
    - v2 (--scale-dtype fp16, default): [K int8 weights][K/32 * 2B fp16 scales]
      For Qwen3-0.6B (K in {1024, 2048, 3072}), row strides are {1088, 2176, 3264} B.
    - v1 (--scale-dtype fp32):          [K int8 weights][K/32 * 4B fp32 scales]
      For Qwen3-0.6B (K in {1024, 2048, 3072}), row strides are {1152, 2304, 3456} B.
  Every row stride is an exact multiple of 64 bytes, ensuring 32-byte aligned AVX2 loads.

v3 is written for "qwen35" (Qwen3.5 hybrid: Gated DeltaNet + gated full attention, e.g. 2B). Full byte-level spec:
docs/qwen35-2b/FORMAT_V3.md. Every tensor start is 64-byte aligned. Descriptor slot table
(slot i = u64 absolute file offset at DESC_OFFS + 8*i; unused slots are 0):

  slot | all types         | type 0 FULL_ATTN                          | type 1 LINEAR_ATTN (DeltaNet, qwen35 only)
  -----+-------------------+-------------------------------------------+-------------------------------------------
   0   | in_norm  f32[dim] |                                           |
   1   | post_norm f32[dim]|                                           |
   2   | W_proj Q8         | rows [q | k | v] (qwen3: 4096 rows);       | rows [qkv(6144) | z(2048)] = 8192 rows
       |                   | qwen35: [q 2048 | k 512 | v 512 | gate 2048] |   (q|k|v each head-major [16][128])
       |                   | = 5120 rows (gate de-interleaved from HF)  |
   3   | W_out Q8          | o_proj [dim, n_heads*head_dim]            | out_proj [2048, 2048]
   4   | W_gateup Q8       | rows [gate_proj | up_proj] (2*hidden rows), identical for every type
   5   | W_down Q8         | down_proj [dim, hidden]
   6   |                   | q_norm f32[head_dim] (qwen35: 1+w baked)  | conv_w f32 TAP-MAJOR [4][6144] (tap 3 = current token)
   7   |                   | k_norm f32[head_dim] (qwen35: 1+w baked)  | negExpA f32[16] = -exp(A_log)
   8   |                   |                                           | dt_bias f32[16]
   9   |                   |                                           | gnorm_w f32[128] (plain w, NOT 1+w)
   10  |                   |                                           | W_b f32 [16][2048] row-major (beta input, NOT quantized)
   11  |                   |                                           | W_a f32 [16][2048] row-major (decay input, NOT quantized)
  12-15 reserved (0)
  Layer types: 0 = full attention (Qwen3 plain, or Qwen3.5 output-gate + partial rotary, selected by flags2/rot_dim),
               1 = Gated DeltaNet. HDR_LAYER_TYPE_MASK bit l = 1 -> layer l is type 1.
  Tokenizer ids needed by the asm are stored in header fields at 176.. (zero = legacy Qwen3 constants), see HDR_TOK_*.
"""

import argparse
import json
import os
import struct
import time
import numpy as np

MAGIC = b"ASML"
VERSION_V1 = 1
VERSION_V2 = 2
VERSION_V3 = 3
VERSION_V4 = 4  # ringzero (M7): v3 + per-slot quant type (Q4_0) + untied lm_head
QTYPE_Q8_0, QTYPE_Q4_0 = 1, 2  # 2-bit descriptor slot types (0 = f32 / unused)
Q4_ROLES = ("attn", "attn_out", "dn", "dn_out", "mlp", "mlp_down")
FLAG_TIED_EMB = 1
FLAG_QK_NORM = 2
FLAG_FP16_SCALES = 4
FLAG2_ATTN_OUT_GATE = 1
FLAG2_NORMS_1PW_BAKED = 2
LAYER_TYPE_FULL_ATTN = 0
LAYER_TYPE_LINEAR_ATTN = 1
HEADER_SIZE = 128
HEADER_SIZE_V3 = 256
DESC_STRIDE = 136  # u32 type, u32 reserved, 16 x u64 absolute file offsets
N_DESC_SLOTS = 16
TOK_HEADER_SIZE = 64
HT_BITS = 19
HT_SIZE = 1 << HT_BITS  # 524,288 slots for 151,387 merges (load factor ~0.289)
HT_MASK = HT_SIZE - 1
FIB_MULT = 0x9E3779B97F4A7C15


def align64(n: int) -> int:
    return (n + 63) & ~63


def pad64(f) -> int:
    pos = f.tell()
    aligned = align64(pos)
    if aligned > pos:
        f.write(b"\x00" * (aligned - pos))
    return aligned


def bytes_to_unicode():
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


def load_safetensors_header(path: str):
    with open(path, "rb") as f:
        hdr_len = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(hdr_len))
    data_offset = 8 + hdr_len
    return hdr, data_offset


def read_tensor_f32(f, hdr, data_offset: int, name: str, r0: int = 0, r1: int = None) -> np.ndarray:
    """Read tensor `name` as float32; optional row range [r0, r1) over dim 0 (ranged file read, no full load)."""
    meta = hdr[name]
    dtype = meta["dtype"]
    shape = tuple(meta["shape"])
    start, end = meta["data_offsets"]
    itemsize = {"BF16": 2, "F16": 2, "F32": 4}.get(dtype)
    if itemsize is None:
        raise ValueError(f"Unsupported dtype {dtype} for tensor {name}")
    if r1 is not None:
        row_bytes = (end - start) // shape[0]
        start, end = start + r0 * row_bytes, start + r1 * row_bytes
        shape = (r1 - r0,) + shape[1:]
    f.seek(data_offset + start)
    raw = f.read(end - start)
    if dtype == "BF16":
        u16 = np.frombuffer(raw, dtype=np.uint16)
        f32 = (u16.astype(np.uint32) << 16).view(np.float32)
        return f32.reshape(shape)
    elif dtype == "F32":
        return np.frombuffer(raw, dtype=np.float32).reshape(shape).copy()
    return np.frombuffer(raw, dtype=np.float16).astype(np.float32).reshape(shape)


def quantize_matrix_q8_0_row_split(w: np.ndarray, scale_dtype: str = "fp16") -> bytes:
    """Quantize [M, K] float32 matrix into per-row split Q8_0 bytes."""
    assert w.ndim == 2
    M, K = w.shape
    assert K % 32 == 0, f"K={K} must be a multiple of 32"
    nb = K // 32
    blocks = w.reshape(M, nb, 32).astype(np.float32, copy=False)
    amax = np.max(np.abs(blocks), axis=2)  # [M, nb]
    scale_f32 = np.where(amax > 0.0, amax / np.float32(127.0), np.float32(0.0)).astype(np.float32)
    if scale_dtype == "fp16":
        scale_f16 = scale_f32.astype(np.float16)
        scale_eff = scale_f16.astype(np.float32)
        inv_scale = np.where(scale_eff > 0.0, np.float32(1.0) / scale_eff, np.float32(0.0)).astype(np.float32)
        q = np.clip(np.rint(blocks * inv_scale[:, :, None]), -127, 127).astype(np.int8)
        s_bytes = scale_f16.astype("<f2").view(np.uint8).reshape(M, nb * 2)
        scale_bytes_per_block = 2
    elif scale_dtype == "fp32":
        inv_scale = np.where(scale_f32 > 0.0, np.float32(1.0) / scale_f32, np.float32(0.0)).astype(np.float32)
        q = np.clip(np.rint(blocks * inv_scale[:, :, None]), -127, 127).astype(np.int8)
        s_bytes = scale_f32.astype("<f4").view(np.uint8).reshape(M, nb * 4)
        scale_bytes_per_block = 4
    else:
        raise ValueError(f"Unsupported scale_dtype: {scale_dtype}")
    q_flat = q.reshape(M, K)  # [M, K] int8
    # Pack each row contiguously: [K int8][nb*scale_bytes_per_block scale bytes]
    row_packed = np.concatenate([q_flat.view(np.uint8), s_bytes], axis=1)
    assert row_packed.shape == (M, K + nb * scale_bytes_per_block)
    return row_packed.tobytes()


def quantize_q4_0_blocks(w: np.ndarray):
    """ringzero (M7): Q4_0 RTN of [M, K] f32 (llama.cpp quantize_row_q4_0_ref rule, block 32): d = (signed absmax) / -8,
    q = clamp(floor(x / d + 8.5), 0, 15), d stored as fp16 and the effective fp16 d used for 1/d; blocks whose fp16 d
    underflows to 0 are flushed to 0. Returns (q - 8 as int8 [M, K/32, 32] in [-8, 7], d f16 [M, K/32])."""
    M, K = w.shape
    assert K % 32 == 0, f"K={K} must be a multiple of 32"
    blocks = w.reshape(M, K // 32, 32).astype(np.float32, copy=False)
    idx = np.argmax(np.abs(blocks), axis=2)
    mx = np.take_along_axis(blocks, idx[:, :, None], axis=2)[:, :, 0]
    d16 = (mx / np.float32(-8.0)).astype(np.float16)
    d = d16.astype(np.float32)
    with np.errstate(divide="ignore"):
        inv = np.where(d != 0.0, np.float32(1.0) / d, np.float32(0.0)).astype(np.float32)
    q = np.clip(np.floor(blocks * inv[:, :, None] + np.float32(8.5)), 0, 15).astype(np.int8) - np.int8(8)
    q[d == 0.0] = 0
    return q, d16


def quantize_matrix_q4_0_row_split(w: np.ndarray) -> bytes:
    """ringzero (M7): per-row split Q4_0: [K/2 nibble bytes][K/32 fp16 d] per row (row stride K/2 + K/16, a 64-byte
    multiple for K % 512 == 0). In each 16-byte block byte j holds element j in the low nibble and element j + 16 in the
    high nibble, both stored as q + 8 (llama.cpp block_q4_0 order)."""
    M, K = w.shape
    q, d16 = quantize_q4_0_blocks(w)
    u = (q + np.int8(8)).astype(np.uint8)
    nib = (u[:, :, :16] | (u[:, :, 16:] << 4)).reshape(M, K // 2)
    row = np.concatenate([nib, d16.astype("<f2").view(np.uint8).reshape(M, K // 16)], axis=1)
    return row.tobytes()


def tokenizer_ids(tok_json_path: str, tok_cfg_path: str) -> dict:
    """Special/chat token ids for the v3 header (qwen35 only) + argmax limit (= number of real token ids)."""
    with open(tok_json_path, "r", encoding="utf-8") as f:
        tj = json.load(f)
    vocab = tj["model"]["vocab"]
    added = {a["content"]: a["id"] for a in tj.get("added_tokens", [])}
    ids = {k: added[v] for k, v in (("endoftext", "<|endoftext|>"), ("im_start", "<|im_start|>"),
                                    ("im_end", "<|im_end|>"), ("think_start", "<think>"), ("think_end", "</think>"))}
    ids.update(user=vocab["user"], assistant=vocab["assistant"], nl=vocab["\u010a"], nl2=vocab["\u010a\u010a"])
    top = max(max(vocab.values()), max(added.values()))
    if os.path.exists(tok_cfg_path):  # tokenizer_config.json declares 7 more special ids (audio/tts) past tokenizer.json
        with open(tok_cfg_path, "r", encoding="utf-8") as f:
            top = max([top] + [int(k) for k in json.load(f).get("added_tokens_decoder", {})])
    ids["argmax_limit"] = top + 1
    return ids


def build_tokenizer_blob(tok_json_path: str, vocab_size: int, tok_cfg_path: str = "") -> bytes:
    with open(tok_json_path, "r", encoding="utf-8") as f:
        tj = json.load(f)

    vocab = tj["model"]["vocab"]
    merges = tj["model"]["merges"]
    added_tokens = tj.get("added_tokens", [])

    b2u = bytes_to_unicode()
    u2b = {v: k for k, v in b2u.items()}

    # 1. byte_to_tok[256]
    byte_to_tok = np.array([vocab[b2u[b]] for b in range(256)], dtype=np.uint32)

    # 2. Raw decoded UTF-8 byte string for every token ID 0..vocab_size-1
    tok_raw = [b""] * vocab_size
    for piece, tid in vocab.items():
        if tid < vocab_size:
            tok_raw[tid] = bytes(u2b[ch] for ch in piece)
    for at in added_tokens:
        tid = at["id"]
        if tid < vocab_size:
            tok_raw[tid] = at["content"].encode("utf-8")
    if tok_cfg_path and os.path.exists(tok_cfg_path):  # extra specials only in tokenizer_config.json (never overrides)
        with open(tok_cfg_path, "r", encoding="utf-8") as f:
            for k, v in json.load(f).get("added_tokens_decoder", {}).items():
                if int(k) < vocab_size and not tok_raw[int(k)]:
                    tok_raw[int(k)] = v["content"].encode("utf-8")

    tok_offsets = np.zeros(vocab_size + 1, dtype=np.uint32)
    chunks = []
    cur = 0
    for tid in range(vocab_size):
        b = tok_raw[tid]
        tok_offsets[tid] = cur
        chunks.append(b)
        cur += len(b)
    tok_offsets[vocab_size] = cur
    tok_bytes = b"".join(chunks)

    # 3. Pre-built Fibonacci-hashed open-addressing merge table
    # Entry layout (16 bytes): <Q key, <I merged_id, <I rank (0xFFFFFFFF = empty)
    ht_keys = np.zeros(HT_SIZE, dtype=np.uint64)
    ht_merged = np.zeros(HT_SIZE, dtype=np.uint32)
    ht_rank = np.full(HT_SIZE, 0xFFFFFFFF, dtype=np.uint32)

    max_probe = 0
    total_probes = 0
    for rank, m in enumerate(merges):
        if isinstance(m, list):
            a, b = m[0], m[1]
        else:
            a, b = m.split(" ")
        ia = vocab[a]
        ib = vocab[b]
        ic = vocab[a + b]
        key = (ia << 32) | ib
        slot = ((key * FIB_MULT) & 0xFFFFFFFFFFFFFFFF) >> (64 - HT_BITS)
        probes = 1
        while ht_rank[slot] != 0xFFFFFFFF:
            slot = (slot + 1) & HT_MASK
            probes += 1
        ht_keys[slot] = key
        ht_merged[slot] = ic
        ht_rank[slot] = rank
        total_probes += probes
        if probes > max_probe:
            max_probe = probes

    print(
        f"Tokenizer: {len(merges)} merges in {HT_SIZE} slots "
        f"(load={len(merges)/HT_SIZE:.3f}, avg_probe={total_probes/len(merges):.2f}, max_probe={max_probe}), "
        f"tok_bytes={len(tok_bytes):,} B"
    )

    # Pack structured array for merge table
    ht_dtype = np.dtype([("key", "<u8"), ("merged_id", "<u4"), ("rank", "<u4")])
    ht_arr = np.empty(HT_SIZE, dtype=ht_dtype)
    ht_arr["key"] = ht_keys
    ht_arr["merged_id"] = ht_merged
    ht_arr["rank"] = ht_rank
    ht_bytes = ht_arr.tobytes()

    # Lay out sub-sections inside tokenizer blob (all 64-byte aligned relative to blob start)
    off_byte_to_tok = TOK_HEADER_SIZE
    off_tok_offsets = align64(off_byte_to_tok + byte_to_tok.nbytes)
    off_tok_bytes = align64(off_tok_offsets + tok_offsets.nbytes)
    off_merge_ht = align64(off_tok_bytes + len(tok_bytes))
    total_tok_size = align64(off_merge_ht + len(ht_bytes))

    buf = bytearray(total_tok_size)
    struct.pack_into(
        "<QQQQ",
        buf,
        0,
        off_byte_to_tok,
        off_tok_offsets,
        off_tok_bytes,
        off_merge_ht,
    )
    buf[off_byte_to_tok : off_byte_to_tok + byte_to_tok.nbytes] = byte_to_tok.tobytes()
    buf[off_tok_offsets : off_tok_offsets + tok_offsets.nbytes] = tok_offsets.tobytes()
    buf[off_tok_bytes : off_tok_bytes + len(tok_bytes)] = tok_bytes
    buf[off_merge_ht : off_merge_ht + len(ht_bytes)] = ht_bytes
    return bytes(buf)


def export_model(model_dir: str, out_path: str, max_seq_len: int = 4096, scale_dtype: str = "fp16"):
    t0 = time.time()
    use_fp16 = scale_dtype == "fp16"
    version = VERSION_V2 if use_fp16 else VERSION_V1
    with open(os.path.join(model_dir, "config.json"), "r") as f:
        cfg = json.load(f)

    dim = int(cfg["hidden_size"])
    hidden_dim = int(cfg["intermediate_size"])
    n_layers = int(cfg["num_hidden_layers"])
    n_heads = int(cfg["num_attention_heads"])
    n_kv_heads = int(cfg["num_key_value_heads"])
    head_dim = int(cfg["head_dim"])
    vocab_size = int(cfg["vocab_size"])
    rms_norm_eps = float(cfg["rms_norm_eps"])
    rope_theta = float(cfg["rope_theta"])
    tie_emb = bool(cfg.get("tie_word_embeddings", True))

    st_path = os.path.join(model_dir, "model.safetensors")
    tok_path = os.path.join(model_dir, "tokenizer.json")
    hdr, data_offset = load_safetensors_header(st_path)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    tok_blob = build_tokenizer_blob(tok_path, vocab_size)

    with open(st_path, "rb") as sf, open(out_path, "wb") as out:
        # Reserve header
        out.write(b"\x00" * HEADER_SIZE)

        # 1. Tokenizer section
        off_tokenizer = pad64(out)
        out.write(tok_blob)

        # 2. Embeddings (tied with lm_head)
        off_embed = pad64(out)
        emb = read_tensor_f32(sf, hdr, data_offset, "model.embed_tokens.weight")
        assert emb.shape == (vocab_size, dim), f"Unexpected embed shape {emb.shape}"
        if "lm_head.weight" in hdr and tie_emb:
            lmh = read_tensor_f32(sf, hdr, data_offset, "lm_head.weight")
            max_diff = float(np.max(np.abs(emb - lmh)))
            assert max_diff == 0.0, f"tie_word_embeddings=True but diff={max_diff}"
            print("Verified embed_tokens == lm_head exact (max_diff=0.0)")
            del lmh
        print(f"Quantizing embed_tokens {emb.shape} -> Q8_0 (scale_dtype={scale_dtype}, v{version})...")
        out.write(quantize_matrix_q8_0_row_split(emb, scale_dtype=scale_dtype))
        del emb

        # 3. Transformer layers (fixed 64-byte aligned stride)
        off_layers = pad64(out)
        layer_stride = 0
        for l in range(n_layers):
            layer_start = pad64(out)
            prefix = f"model.layers.{l}"

            # 4 FP32 norm vectors
            in_norm = read_tensor_f32(sf, hdr, data_offset, f"{prefix}.input_layernorm.weight")
            q_norm = read_tensor_f32(sf, hdr, data_offset, f"{prefix}.self_attn.q_norm.weight")
            k_norm = read_tensor_f32(sf, hdr, data_offset, f"{prefix}.self_attn.k_norm.weight")
            post_norm = read_tensor_f32(sf, hdr, data_offset, f"{prefix}.post_attention_layernorm.weight")

            assert in_norm.shape == (dim,)
            assert q_norm.shape == (head_dim,)
            assert k_norm.shape == (head_dim,)
            assert post_norm.shape == (dim,)

            out.write(in_norm.astype("<f4").tobytes())
            out.write(q_norm.astype("<f4").tobytes())
            out.write(k_norm.astype("<f4").tobytes())
            out.write(post_norm.astype("<f4").tobytes())

            # 7 Q8_0 projection matrices in fixed order
            proj_specs = [
                (f"{prefix}.self_attn.q_proj.weight", (n_heads * head_dim, dim)),
                (f"{prefix}.self_attn.k_proj.weight", (n_kv_heads * head_dim, dim)),
                (f"{prefix}.self_attn.v_proj.weight", (n_kv_heads * head_dim, dim)),
                (f"{prefix}.self_attn.o_proj.weight", (dim, n_heads * head_dim)),
                (f"{prefix}.mlp.gate_proj.weight", (hidden_dim, dim)),
                (f"{prefix}.mlp.up_proj.weight", (hidden_dim, dim)),
                (f"{prefix}.mlp.down_proj.weight", (dim, hidden_dim)),
            ]
            for name, expected_shape in proj_specs:
                pad64(out)
                mat = read_tensor_f32(sf, hdr, data_offset, name)
                assert mat.shape == expected_shape, f"{name}: got {mat.shape}, expected {expected_shape}"
                out.write(quantize_matrix_q8_0_row_split(mat, scale_dtype=scale_dtype))

            layer_end = pad64(out)
            this_stride = layer_end - layer_start
            if l == 0:
                layer_stride = this_stride
            else:
                assert this_stride == layer_stride, f"Layer {l} stride {this_stride} != {layer_stride}"

            if (l + 1) % 7 == 0 or l == n_layers - 1:
                print(f"  Exported layer {l + 1}/{n_layers} (stride={layer_stride:,} B)")

        # 4. Final RMSNorm
        off_final_norm = pad64(out)
        final_norm = read_tensor_f32(sf, hdr, data_offset, "model.norm.weight")
        assert final_norm.shape == (dim,)
        out.write(final_norm.astype("<f4").tobytes())
        file_size = pad64(out)

        # Write 128-byte header
        flags = (FLAG_TIED_EMB if tie_emb else 0) | FLAG_QK_NORM | (FLAG_FP16_SCALES if use_fp16 else 0)
        eos_1 = 151645  # <|im_end|>
        eos_2 = 151643  # <|endoftext|>
        header = struct.pack(
            "<4sIIIIIIIIIffIIIIQQQQQQ16s",
            MAGIC,
            version,
            dim,
            hidden_dim,
            n_layers,
            n_heads,
            n_kv_heads,
            head_dim,
            vocab_size,
            max_seq_len,
            rms_norm_eps,
            rope_theta,
            eos_1,
            eos_2,
            HT_MASK,
            flags,
            off_tokenizer,
            off_embed,
            off_layers,
            layer_stride,
            off_final_norm,
            file_size,
            b"\x00" * 16,
        )
        assert len(header) == HEADER_SIZE
        out.seek(0)
        out.write(header)

    dt = time.time() - t0
    print(f"\nExport complete: {out_path}")
    print(f"  Format       : .asmbin v{version} (scale_dtype={scale_dtype}, flags=0x{flags:x})")
    print(f"  Total size   : {file_size:,} bytes ({file_size / (1024**2):.2f} MiB)")
    print(f"  off_tokenizer: {off_tokenizer:,}")
    print(f"  off_embed    : {off_embed:,}")
    print(f"  off_layers   : {off_layers:,} (stride={layer_stride:,} x {n_layers})")
    print(f"  off_final_nrm: {off_final_norm:,}")
    print(f"  Elapsed      : {dt:.2f} s")


def is_qwen35(cfg: dict) -> bool:
    tcfg = cfg.get("text_config", cfg)
    layer_types = tcfg.get("layer_types") or []
    return "linear_attention" in layer_types or tcfg.get("model_type", "").startswith("qwen3_5")


def export_model_v3(model_dir: str, out_path: str, max_seq_len: int = 4096, scale_dtype: str = "fp16", force: bool = False,
                    q4: frozenset = frozenset()):
    """ringzero (M7): q4 = set of matrix roles written as Q4_0 (Q4_ROLES); any Q4 tensor or an untied lm_head makes the
    file v4 (descriptor reserved u32 = 2-bit quant type per slot, header u64 216 = lm_head offset). Otherwise v3, byte
    for byte as before."""
    t0 = time.time()
    use_fp16 = scale_dtype == "fp16"
    assert use_fp16 or not q4, "Q4_0 needs fp16 scales"
    with open(os.path.join(model_dir, "config.json"), "r") as f:
        cfg = json.load(f)
    tcfg = cfg.get("text_config", cfg)
    layer_types = tcfg.get("layer_types") or ["full_attention"] * int(tcfg["num_hidden_layers"])
    is35 = is_qwen35(cfg)
    assert is35 or force, "the v3 writer is for Qwen3.5 (Qwen3 exports as v1/v2; ringzero: --v3 forces it)"
    pre = "model.language_model." if is35 else "model."  # ringzero: "" for unprefixed weights, set below
    print(f"Architecture: {'qwen35 (hybrid DeltaNet + gated attention)' if is35 else 'qwen3 (dense full attention)'}")

    dim = int(tcfg["hidden_size"])
    hidden_dim = int(tcfg["intermediate_size"])
    n_layers = int(tcfg["num_hidden_layers"])
    n_heads = int(tcfg["num_attention_heads"])
    n_kv_heads = int(tcfg["num_key_value_heads"])
    head_dim = int(tcfg["head_dim"])
    vocab_size = int(tcfg["vocab_size"])
    rms_norm_eps = float(tcfg["rms_norm_eps"])
    rp = tcfg.get("rope_parameters", {})
    rope_theta = float(rp["rope_theta"] if "rope_theta" in rp else tcfg["rope_theta"])
    rot_dim = int(head_dim * float(rp.get("partial_rotary_factor", 1.0)))
    tie_emb = bool(cfg.get("tie_word_embeddings", tcfg.get("tie_word_embeddings", True)))
    n_lk, n_lv = int(tcfg.get("linear_num_key_heads", 0)), int(tcfg.get("linear_num_value_heads", 0))
    lk_dim, lv_dim = int(tcfg.get("linear_key_head_dim", 0)), int(tcfg.get("linear_value_head_dim", 0))
    conv_k = int(tcfg.get("linear_conv_kernel_dim", 0))
    gate = bool(tcfg.get("attn_output_gate", False)) and is35
    one_pw = 1.0 if is35 else 0.0  # Qwen3.5 RMSNorm is x*(1+w): bake +1.0 (all but the DeltaNet gated norm)
    if is35:  # ringzero (M7): Hv may be a multiple of Hk (9B: 16 K heads, 32 V heads; V head j uses K head j // (Hv/Hk))
        assert n_lv % n_lk == 0 and lk_dim == lv_dim, "need Hv % Hk == 0 and Dk == Dv"
    version = VERSION_V4 if (q4 or not tie_emb) else VERSION_V3
    mask = sum(1 << l for l, t in enumerate(layer_types) if t == "linear_attention")
    assert n_layers <= 64

    wm = os.path.join(model_dir, "model.safetensors.index.json")
    shard_of = None  # ringzero (M7): multi-shard checkpoints (9B: 4 shards), tensor name -> shard file
    st_name = "model.safetensors"
    if os.path.exists(wm):
        shard_of = json.load(open(wm))["weight_map"]
        st_name = sorted(set(shard_of.values()))[0]
    st_path = os.path.join(model_dir, st_name)
    tok_path = os.path.join(model_dir, "tokenizer.json")
    tok_cfg = os.path.join(model_dir, "tokenizer_config.json")
    shards = {}
    for fn in sorted(set(shard_of.values())) if shard_of else [st_name]:
        h, o = load_safetensors_header(os.path.join(model_dir, fn))
        shards[fn] = (open(os.path.join(model_dir, fn), "rb"), h, o)
    hdr = {k: v for _, h, _ in shards.values() for k, v in h.items()}
    if not is35 and pre + "embed_tokens.weight" not in hdr and "embed_tokens.weight" in hdr:
        pre = ""  # ringzero: Qwen3-Embedding stores the bare Qwen3Model (no "model." prefix)
    if is35 and pre + "embed_tokens.weight" not in hdr and "model.embed_tokens.weight" in hdr:
        pre = "model."  # ringzero (T59): text-only Qwen3.5 repacks (Qwen3_5ForCausalLM, NeoHorse-1-4B) drop "language_model."

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    tok_blob = build_tokenizer_blob(tok_path, vocab_size, tok_cfg)
    tids = tokenizer_ids(tok_path, tok_cfg) if is35 else None

    def RT(name, r0=0, r1=None):
        sf, h, o = shards[shard_of[name]] if shard_of else shards[st_name]
        return read_tensor_f32(sf, h, o, name, r0, r1)

    with open(out_path, "wb") as out:
        def T(name, r0=0, r1=None):
            return RT(pre + name, r0, r1)

        def put_f32(arr, shape):
            arr = np.asarray(arr, dtype=np.float32)
            assert arr.shape == shape, f"got {arr.shape}, expected {shape}"
            off = pad64(out)
            out.write(arr.astype("<f4").tobytes())
            return off

        def put_q8(mats, K, role=None):
            """Quantize and write [M_i, K] matrices back to back (rows chunked: bounded memory). Returns start offset.
            ringzero (M7): Q4_0 when role is in q4 (qtypes[] records the slot type for the descriptor)."""
            off = pad64(out)
            four = role in q4
            for m in mats:
                assert m.ndim == 2 and m.shape[1] == K and K % 32 == 0 and K <= 16384, m.shape
                assert not four or K % 512 == 0, K  # Q4 row stride K/2 + K/16 stays a 64-byte multiple
                for r in range(0, m.shape[0], 8192):
                    c = m[r:r + 8192]
                    out.write(quantize_matrix_q4_0_row_split(c) if four else quantize_matrix_q8_0_row_split(c, scale_dtype=scale_dtype))
            qtypes.append(QTYPE_Q4_0 if four else QTYPE_Q8_0)
            return off

        def q8_tensor(name, shape):
            m = T(name)
            assert m.shape == shape, f"{name}: got {m.shape}, expected {shape}"
            return m

        # Reserve header
        out.write(b"\x00" * HEADER_SIZE_V3)

        # 0. Reserve the per-layer descriptor table (filled in once all offsets are known)
        off_layer_table = pad64(out)
        out.write(b"\x00" * (DESC_STRIDE * n_layers))

        # 1. Tokenizer section
        off_tokenizer = pad64(out)
        out.write(tok_blob)

        # 2. Embeddings (tied with lm_head), streamed in row chunks
        off_embed = pad64(out)
        ename = "embed_tokens.weight"
        assert tuple(hdr[pre + ename]["shape"]) == (vocab_size, dim), f"Unexpected embed shape {hdr[pre + ename]['shape']}"
        if "lm_head.weight" in hdr and tie_emb:
            emb = T(ename)
            lmh = RT("lm_head.weight")
            max_diff = float(np.max(np.abs(emb - lmh)))
            assert max_diff == 0.0, f"tie_word_embeddings=True but diff={max_diff}"
            print("Verified embed_tokens == lm_head exact (max_diff=0.0)")
            del emb, lmh
        print(f"Quantizing embed_tokens ({vocab_size}, {dim}) -> Q8_0 (scale_dtype={scale_dtype}, v{version})...")
        for r in range(0, vocab_size, 8192):
            out.write(quantize_matrix_q8_0_row_split(T(ename, r, min(r + 8192, vocab_size)), scale_dtype=scale_dtype))

        # 3. Transformer layers: each tensor 64-byte aligned, located only through the descriptor table
        off_layers = pad64(out)
        descs = []
        for l in range(n_layers):
            lin = layer_types[l] == "linear_attention"
            p = f"layers.{l}."
            s = [0] * N_DESC_SLOTS
            qtypes = []  # appended by put_q8 for slots 2, 3, 4, 5 in that order
            # file order: small fp32 tensors first (Qwen3 order: in_norm, q_norm, k_norm, post_norm), then Q8 matrices
            s[0] = put_f32(T(p + "input_layernorm.weight") + one_pw, (dim,))
            if not lin:
                a = p + "self_attn."
                s[6] = put_f32(T(a + "q_norm.weight") + one_pw, (head_dim,))
                s[7] = put_f32(T(a + "k_norm.weight") + one_pw, (head_dim,))
                ltype = LAYER_TYPE_FULL_ATTN
            else:
                a = p + "linear_attn."
                C = 2 * n_lk * lk_dim + n_lv * lv_dim  # 6144 conv channels (q|k|v)
                zdim = n_lv * lv_dim
                cw = T(a + "conv1d.weight")
                assert cw.shape == (C, 1, conv_k)
                s[6] = put_f32(np.ascontiguousarray(cw[:, 0, :].T), (conv_k, C))  # tap-major [4][C]
                s[7] = put_f32((-np.exp(T(a + "A_log").astype(np.float64))).astype(np.float32), (n_lv,))
                s[8] = put_f32(T(a + "dt_bias"), (n_lv,))
                s[9] = put_f32(T(a + "norm.weight"), (lv_dim,))  # plain weight (gated RMSNorm), no +1
                s[10] = put_f32(T(a + "in_proj_b.weight"), (n_lv, dim))
                s[11] = put_f32(T(a + "in_proj_a.weight"), (n_lv, dim))
                ltype = LAYER_TYPE_LINEAR_ATTN
            s[1] = put_f32(T(p + "post_attention_layernorm.weight") + one_pw, (dim,))
            if not lin:
                q = q8_tensor(a + "q_proj.weight", (n_heads * head_dim * (2 if gate else 1), dim))
                if gate:  # HF per-head interleave [q_h(256) | gate_h(256)] -> [all q rows | all gate rows]
                    q = q.reshape(n_heads, 2, head_dim, dim)
                    q, g = q[:, 0].reshape(-1, dim), q[:, 1].reshape(-1, dim)
                k = q8_tensor(a + "k_proj.weight", (n_kv_heads * head_dim, dim))
                v = q8_tensor(a + "v_proj.weight", (n_kv_heads * head_dim, dim))
                s[2] = put_q8([q, k, v] + ([g] if gate else []), dim, "attn")
                s[3] = put_q8([q8_tensor(a + "o_proj.weight", (dim, n_heads * head_dim))], n_heads * head_dim, "attn_out")
            else:
                s[2] = put_q8([q8_tensor(a + "in_proj_qkv.weight", (C, dim)),
                               q8_tensor(a + "in_proj_z.weight", (zdim, dim))], dim, "dn")
                s[3] = put_q8([q8_tensor(a + "out_proj.weight", (dim, zdim))], zdim, "dn_out")
            s[4] = put_q8([q8_tensor(p + "mlp.gate_proj.weight", (hidden_dim, dim)),
                           q8_tensor(p + "mlp.up_proj.weight", (hidden_dim, dim))], dim, "mlp")
            s[5] = put_q8([q8_tensor(p + "mlp.down_proj.weight", (dim, hidden_dim))], hidden_dim, "mlp_down")
            pad64(out)
            qbits = sum(t << (2 * i) for i, t in zip((2, 3, 4, 5), qtypes)) if version == VERSION_V4 else 0
            descs.append(struct.pack("<II16Q", ltype, qbits, *s))
            assert len(descs[-1]) == DESC_STRIDE
            if (l + 1) % 7 == 0 or l == n_layers - 1:
                print(f"  Exported layer {l + 1}/{n_layers}", flush=True)

        # Qwen3 layers are uniform: report the stride (informational only); hybrid layers are not
        offs_l = [struct.unpack_from("<Q", d, 8)[0] for d in descs]
        strides = {b - a for a, b in zip(offs_l, offs_l[1:])}
        layer_stride = strides.pop() if len(strides) == 1 and not is35 else 0

        # 4. Final RMSNorm
        off_final_norm = pad64(out)
        final_norm = T("norm.weight") + one_pw
        assert final_norm.shape == (dim,)
        out.write(final_norm.astype("<f4").tobytes())
        off_lm_head = 0
        if not tie_emb:  # ringzero (M7): untied lm_head (9B), Q8_0, streamed in row chunks
            off_lm_head = pad64(out)
            print(f"Quantizing lm_head ({vocab_size}, {dim}) -> Q8_0...")
            for r in range(0, vocab_size, 8192):
                out.write(quantize_matrix_q8_0_row_split(RT("lm_head.weight", r, min(r + 8192, vocab_size)), scale_dtype=scale_dtype))
        file_size = pad64(out)

        out.seek(off_layer_table)
        out.write(b"".join(descs))

        # Write 128-byte header
        flags = (FLAG_TIED_EMB if tie_emb else 0) | FLAG_QK_NORM | (FLAG_FP16_SCALES if use_fp16 else 0)
        eos_1, eos_2 = (tids["im_end"], tids["endoftext"]) if is35 else (151645, 151643)  # <|im_end|>, <|endoftext|>
        header = struct.pack(
            "<4sIIIIIIIIIffIIIIQQQQQQ16s",
            MAGIC,
            version,
            dim,
            hidden_dim,
            n_layers,
            n_heads,
            n_kv_heads,
            head_dim,
            vocab_size,
            max_seq_len,
            rms_norm_eps,
            rope_theta,
            eos_1,
            eos_2,
            HT_MASK,
            flags,
            off_tokenizer,
            off_embed,
            off_layers,
            layer_stride,
            off_final_norm,
            file_size,
            b"\x00" * 16,
        )
        assert len(header) == 128
        # v3 extension (offsets 128..): layer table, type map, linear-attn dims, rotary, flags2
        header += struct.pack(
            "<QQIIIIIIII",
            off_layer_table,
            mask,  # layer_type_mask: bit l = 1 -> layer l is not type 0 (DeltaNet)
            n_lk, n_lv, lk_dim, lv_dim,
            conv_k,
            rot_dim,  # Qwen3: head_dim (full rotary); Qwen3.5: 64
            (FLAG2_ATTN_OUT_GATE if gate else 0) | (FLAG2_NORMS_1PW_BAKED if is35 else 0),
            0,
        )
        assert len(header) == 176  # 172..175 pad (the trailing 0 u32)
        if is35:  # tokenizer ids (HDR_TOK_* at 176..215); all-zero for qwen3 = use the legacy constants.inc ids
            header += struct.pack("<10I", tids["endoftext"], tids["im_start"], tids["im_end"], tids["think_start"],
                                  tids["think_end"], tids["user"], tids["assistant"], tids["nl"], tids["nl2"],
                                  tids["argmax_limit"])
        if version == VERSION_V4:  # ringzero (M7): 216 u64 lm_head offset (0 = tied to embed_tokens)
            header += b"\x00" * (216 - len(header)) + struct.pack("<Q", off_lm_head)
        header += b"\x00" * (HEADER_SIZE_V3 - len(header))
        assert len(header) == HEADER_SIZE_V3
        out.seek(0)
        out.write(header)

    dt = time.time() - t0
    print(f"\nExport complete: {out_path}")
    print(f"  Format       : .asmbin v{version} (scale_dtype={scale_dtype}, flags=0x{flags:x})")
    print(f"  Total size   : {file_size:,} bytes ({file_size / (1024**2):.2f} MiB)")
    print(f"  off_tokenizer: {off_tokenizer:,}")
    print(f"  off_embed    : {off_embed:,}")
    print(f"  off_layertbl : {off_layer_table:,} ({n_layers} x {DESC_STRIDE} B)")
    print(f"  off_layers   : {off_layers:,} (stride={layer_stride:,} x {n_layers})")
    print(f"  off_final_nrm: {off_final_norm:,}")
    print(f"  Q4_0 roles   : {','.join(sorted(q4)) or '-'}; lm_head {'tied' if tie_emb else 'Q8_0 untied'}")
    print(f"  Elapsed      : {dt:.2f} s")


def main():
    ap = argparse.ArgumentParser(description="Export Qwen3 / Qwen3.5 safetensors to .asmbin Q8_0")
    ap.add_argument("--model-dir", default="models/qwen3-0.6b-bf16", help="Directory containing config.json, model.safetensors, tokenizer.json")
    ap.add_argument("--out", default="models/qwen3-0.6b-q8_0.asmbin", help="Output .asmbin path")
    ap.add_argument("--max-seq-len", type=int, default=4096)
    ap.add_argument("--scale-dtype", choices=["fp16", "fp32"], default="fp16", help="Per-block Q8_0 scale dtype (fp16 -> v2 compact, fp32 -> v1 legacy)")
    ap.add_argument("--v3", action="store_true", help="ringzero: write a Qwen3 checkpoint as v3 (for the QWEN35 kernel build)")
    ap.add_argument("--q4", default="", help="ringzero (M7): Q4_0 matrix roles, comma list of " + ",".join(Q4_ROLES) +
                    " or 'all' (embeddings and lm_head stay Q8_0, small tensors f32); writes .asmbin v4")
    args = ap.parse_args()
    q4 = frozenset(Q4_ROLES if args.q4 == "all" else [r for r in args.q4.split(",") if r])
    assert q4 <= set(Q4_ROLES), f"unknown --q4 roles {sorted(q4 - set(Q4_ROLES))}"
    with open(os.path.join(args.model_dir, "config.json"), "r") as f:
        cfg = json.load(f)
    if is_qwen35(cfg) or args.v3:  # Qwen3 keeps the original v1/v2 writer unless --v3
        export_model_v3(args.model_dir, args.out, args.max_seq_len, scale_dtype=args.scale_dtype, force=args.v3, q4=q4)
    else:
        export_model(args.model_dir, args.out, args.max_seq_len, scale_dtype=args.scale_dtype)


if __name__ == "__main__":
    main()
