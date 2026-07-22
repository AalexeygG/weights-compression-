import numpy as np
import zstandard

from s1p2_io import decode_s1p2


def sparsify(nibbles: np.ndarray, sparsity_target_pct: float) -> tuple[np.ndarray, np.ndarray]:
    # rank-based, not percentile - percentile breaks on discrete 0-15 values
    # importance = magnitude of the decoded value, not the raw nibble code
    # (nibble codes are sign+magnitude, so nibble value isn't ordered by real value)
    magnitude = np.abs(decode_s1p2(nibbles))
    n_values = len(nibbles)
    n_to_zero = int(n_values * sparsity_target_pct / 100)

    order = np.argsort(magnitude, kind='stable')
    sparse_mask = np.zeros(n_values, dtype=bool)
    sparse_mask[order[:n_to_zero]] = True

    sparsified = nibbles.copy()
    sparsified[sparse_mask] = 0
    return sparsified, sparse_mask


def pack_nibbles_to_bytes(nibbles: np.ndarray) -> bytes:
    nibbles = nibbles.astype(np.uint8)
    if len(nibbles) % 2:
        nibbles = np.append(nibbles, 0)
    low = nibbles[0::2]
    high = nibbles[1::2]
    return ((high << 4) | low).tobytes()


def compress_f(nibbles: np.ndarray, sparsity_target_pct: float = 0, level: int = 3) -> tuple[bytes, np.ndarray]:
    if sparsity_target_pct > 0:
        sparsified, sparse_mask = sparsify(nibbles, sparsity_target_pct)
    else:
        sparsified, sparse_mask = nibbles, np.zeros(len(nibbles), dtype=bool)
    packed = pack_nibbles_to_bytes(sparsified)
    compressed = zstandard.ZstdCompressor(level=level).compress(packed)
    return compressed, sparse_mask


def decompress_f(compressed: bytes, n_values: int) -> np.ndarray:
    packed = zstandard.ZstdDecompressor().decompress(compressed)
    data = np.frombuffer(packed, dtype=np.uint8)
    nibbles = np.empty(len(data) * 2, dtype=np.int32)
    nibbles[0::2] = data & 0x0F
    nibbles[1::2] = (data >> 4) & 0x0F
    return nibbles[:n_values]
