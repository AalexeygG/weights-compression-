"""
Extra lossy compression on top of S1P2 (HiF4) weight nibbles.

compress(weight_bytes, in_dim, keep_top_frac, act_norm, mode) -> blob
decompress(blob) -> weight_bytes
"""

import json
from pathlib import Path
import numpy as np
import zstandard


def _unpack(raw):
    data = np.frombuffer(raw, dtype=np.uint8)
    nib = np.empty(len(data) * 2, dtype=np.uint8)
    nib[0::2] = data & 0x0F
    nib[1::2] = data >> 4
    return nib


def _pack(nib):
    nib = nib.astype(np.uint8)
    if len(nib) % 2:
        nib = np.append(nib, 0)
    return ((nib[1::2] << 4) | nib[0::2]).tobytes()


def _decode(nib):
    sign = (nib >> 3) & 1
    mag = ((nib >> 2) & 1) + (nib & 3) * 0.25
    return np.where(sign == 1, -mag, mag).astype(np.float64)


def activation_norm(act_base_path):
    base = str(act_base_path)
    read = lambda s: np.frombuffer(Path(base + s).read_bytes(), dtype=np.uint8)
    meta = json.loads(Path(base + ".json").read_text())
    rows, cols = meta["quantized_shape"]
    nb = meta["num_blocks"]
    vals = _decode(_unpack(read("_S1P2.bin"))[: rows * cols]).reshape(nb, 64)
    e6 = read("_E6M2.bin").astype(np.int64)
    bscale = (2.0 ** ((e6 >> 2) - 48)) * (1 + (e6 & 3) / 4.0)
    bits = lambda a, n: np.stack([(a >> b) & 1 for b in range(n)], axis=1)
    b8 = bits(read("_E1_8.bin"), 8)
    e116 = read("_E1_16.bin").reshape(nb, 2)
    b16 = bits(e116[:, 0].astype(np.uint16) | (e116[:, 1].astype(np.uint16) << 8), 16)
    p = np.arange(64)
    real = bscale[:, None] * 2.0 ** (b8[:, p // 8] + b16[:, p // 4]) * vals
    real = real.reshape(rows, cols)
    return np.sqrt((real ** 2).sum(axis=0))


def compress(weight_bytes, in_dim, keep_top_frac=0.5, act_norm=None, mode="mag", level=3):
    nib = _unpack(weight_bytes)
    if mode == "mag" or act_norm is None:
        score = np.abs(_decode(nib))
    else:
        a = act_norm[np.arange(len(nib)) // in_dim]
        score = a if mode == "act" else np.abs(_decode(nib)) * a
    if keep_top_frac < 1.0:
        # rank-based, not threshold - threshold misses the target % on tied scores
        n_protect = int(len(nib) * keep_top_frac)
        order = np.argsort(score, kind="stable")
        protect = np.zeros(len(nib), dtype=bool)
        protect[order[len(nib) - n_protect:]] = True
        k = nib & 7
        sign = nib >> 3
        k = np.where((k % 2 == 1) & ~protect, k - 1, k)
        nib = np.where(k == 0, 0, (sign << 3) | k).astype(np.uint8)
    return zstandard.ZstdCompressor(level=level).compress(_pack(nib))


def decompress(blob):
    return zstandard.ZstdDecompressor().decompress(blob)
