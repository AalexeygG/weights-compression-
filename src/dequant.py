import json
import numpy as np
from pathlib import Path

from s1p2_io import decode_s1p2

# HiF4 dequant (arXiv 2602.11287v2, Eq. 2): block of 64 elements,
# V_i = E6M2 * 2^(E1_8[i//8] + E1_16[i//4]) * decode_s1p2(nibble_i)


def _bitcols(vals: np.ndarray, nbits: int) -> np.ndarray:
    return np.stack([(vals >> b) & 1 for b in range(nbits)], axis=1)


def dequantize(base_path: str | Path) -> np.ndarray:
    base = str(base_path)
    read = lambda suf: np.frombuffer(Path(base + suf).read_bytes(), dtype=np.uint8)
    meta = json.loads(Path(base + ".json").read_text())
    rows, cols = meta["quantized_shape"]
    n = rows * cols
    nb = meta["num_blocks"]

    sb = read("_S1P2.bin")
    nib = np.empty(len(sb) * 2, dtype=np.uint8)
    nib[0::2] = sb & 0x0F
    nib[1::2] = sb >> 4
    values = decode_s1p2(nib[:n]).astype(np.float64).reshape(nb, 64)

    e6 = read("_E6M2.bin").astype(np.int64)
    block_scale = (2.0 ** ((e6 >> 2) - 48)) * (1 + (e6 & 3) / 4.0)

    b8 = _bitcols(read("_E1_8.bin"), 8)
    e116 = read("_E1_16.bin").reshape(nb, 2)
    b16 = _bitcols(e116[:, 0].astype(np.uint16) | (e116[:, 1].astype(np.uint16) << 8), 16)

    pos = np.arange(64)
    micro = 2.0 ** (b8[:, pos // 8] + b16[:, pos // 4])
    real = block_scale[:, None] * micro * values
    return real.reshape(rows, cols)
