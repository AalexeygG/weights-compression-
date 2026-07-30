import numpy as np
import zstandard

from s1p2_io import decode_s1p2


# codes are sign+magnitude, so rank by decoded |value|, not by code;
# rank instead of percentile since thresholds misbehave on 16 discrete levels
def sparsify(nibbles: np.ndarray, sparsity_target_pct: float) -> tuple[np.ndarray, np.ndarray]:
    magnitude = np.abs(decode_s1p2(nibbles))
    n_values = len(nibbles)
    n_to_zero = int(n_values * sparsity_target_pct / 100)

    order = np.argsort(magnitude, kind='stable')
    sparse_mask = np.zeros(n_values, dtype=bool)
    sparse_mask[order[:n_to_zero]] = True

    sparsified = nibbles.copy()
    sparsified[sparse_mask] = 0
    return sparsified, sparse_mask


# merge adjacent magnitude levels pairwise (step 0.25 -> 0.5) up to max_magnitude,
# rounding toward zero; fewer distinct values -> better entropy coding, error <= 0.25
def coarsen(nibbles: np.ndarray, max_magnitude: float = 1.75) -> np.ndarray:
    k = nibbles & 7
    sign = nibbles >> 3
    affected = (k * 0.25 <= max_magnitude) & (k % 2 == 1)
    k = np.where(affected, k - 1, k)
    return np.where(k == 0, 0, (sign << 3) | k).astype(nibbles.dtype)


def pack_nibbles_to_bytes(nibbles: np.ndarray) -> bytes:
    nibbles = nibbles.astype(np.uint8)
    if len(nibbles) % 2:
        nibbles = np.append(nibbles, 0)
    low = nibbles[0::2]
    high = nibbles[1::2]
    return ((high << 4) | low).tobytes()


def compress_f(nibbles: np.ndarray, sparsity_target_pct: float = 0,
               coarsen_max: float = 0, level: int = 3) -> tuple[bytes, np.ndarray]:
    if sparsity_target_pct > 0:
        out, sparse_mask = sparsify(nibbles, sparsity_target_pct)
    else:
        out, sparse_mask = nibbles, np.zeros(len(nibbles), dtype=bool)
    if coarsen_max > 0:
        out = coarsen(out, coarsen_max)
    packed = pack_nibbles_to_bytes(out)
    compressed = zstandard.ZstdCompressor(level=level).compress(packed)
    return compressed, sparse_mask


def decompress_f(compressed: bytes, n_values: int) -> np.ndarray:
    packed = zstandard.ZstdDecompressor().decompress(compressed)
    data = np.frombuffer(packed, dtype=np.uint8)
    nibbles = np.empty(len(data) * 2, dtype=np.int32)
    nibbles[0::2] = data & 0x0F
    nibbles[1::2] = (data >> 4) & 0x0F
    return nibbles[:n_values]
