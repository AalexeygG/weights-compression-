"""Extra compression on top of S1P2 (HiF4) nibbles, for weights and activations."""

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


def _top_mask(strength, keep_top_frac):
    k = int(len(strength) * keep_top_frac)
    order = np.argsort(strength, kind="stable")
    m = np.zeros(len(strength), dtype=bool)
    m[order[len(strength) - k:]] = True
    return m


def _protect_mask(nib, keep_top_frac, granularity, row_len, score):
    if granularity == "value":
        return _top_mask(score, keep_top_frac)
    absm = np.abs(_decode(nib)).reshape(-1, row_len)
    if granularity == "channel":
        keep = _top_mask(absm.mean(axis=0), keep_top_frac)
        return np.broadcast_to(keep[None, :], absm.shape).ravel()
    keep = _top_mask(absm.mean(axis=1), keep_top_frac)   # token
    return np.broadcast_to(keep[:, None], absm.shape).ravel()


def _coarsen(nib, protect):
    k = nib & 7
    sign = nib >> 3
    k = np.where((k % 2 == 1) & ~protect, k - 1, k)
    return np.where(k == 0, 0, (sign << 3) | k).astype(np.uint8)


def _dedup_encode(nib, row_len):
    rows = nib.reshape(-1, row_len)
    uniq, inv = np.unique(rows, axis=0, return_inverse=True)
    header = np.array([rows.shape[0], row_len, uniq.shape[0]], dtype=np.uint32)
    return header.tobytes() + inv.astype(np.uint32).ravel().tobytes() + _pack(uniq.ravel())


def _dedup_decode(payload):
    n_rows, row_len, n_uniq = np.frombuffer(payload[:12], dtype=np.uint32)
    off = 12
    index = np.frombuffer(payload[off:off + n_rows * 4], dtype=np.uint32)
    off += n_rows * 4
    uniq = _unpack(payload[off:])[: n_uniq * row_len].reshape(n_uniq, row_len)
    return _pack(uniq[index].ravel())


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
    return np.sqrt((real.reshape(rows, cols) ** 2).sum(axis=0))


def compress(data_bytes, in_dim=0, keep_top_frac=1.0, mode="mag", act_norm=None,
             granularity="value", row_len=0, dedup=False, level=3):
    nib = _unpack(data_bytes)
    if keep_top_frac < 1.0:
        if mode == "mag" or act_norm is None:
            score = np.abs(_decode(nib))
        else:
            a = act_norm[np.arange(len(nib)) // in_dim]
            score = a if mode == "act" else np.abs(_decode(nib)) * a
        protect = _protect_mask(nib, keep_top_frac, granularity, row_len, score)
        nib = _coarsen(nib, protect)
    if dedup and row_len and len(nib) % row_len == 0:
        payload = b"\x01" + _dedup_encode(nib, row_len)
    else:
        payload = b"\x00" + _pack(nib)
    return zstandard.ZstdCompressor(level=level).compress(payload)


def compress_auto(data_bytes, in_dim=0, keep_top_frac=0.5, threshold=1.3, level=3):
    # keep lossless if it already compresses enough, else coarsen
    blob = compress(data_bytes, level=level)
    if len(data_bytes) / len(blob) >= threshold:
        return blob
    return compress(data_bytes, in_dim=in_dim, keep_top_frac=keep_top_frac, level=level)


def decompress(blob):
    payload = zstandard.ZstdDecompressor().decompress(blob)
    if payload[0] == 1:
        return _dedup_decode(payload[1:])
    return payload[1:]
