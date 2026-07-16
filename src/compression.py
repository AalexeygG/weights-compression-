import numpy as np
import heapq
from collections import Counter

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


def build_huffman_codebook(freq_counter: Counter) -> dict:
    heap = [[weight, [symbol, ""]] for symbol, weight in freq_counter.items()]
    heapq.heapify(heap)

    if len(heap) == 1:
        symbol = heap[0][1][0]
        return {symbol: "0"}

    while len(heap) > 1:
        lo = heapq.heappop(heap)
        hi = heapq.heappop(heap)
        for pair in lo[1:]:
            pair[1] = "0" + pair[1]
        for pair in hi[1:]:
            pair[1] = "1" + pair[1]
        heapq.heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])

    return {symbol: code for symbol, code in heap[0][1:]}


def compress_f(nibbles: np.ndarray, sparsity_target_pct: float = 30) -> tuple[str, dict, np.ndarray]:
    sparsified, sparse_mask = sparsify(nibbles, sparsity_target_pct)
    freq = Counter(sparsified.tolist())
    codebook = build_huffman_codebook(freq)
    compressed_bits = "".join(codebook[v] for v in sparsified.tolist())
    return compressed_bits, codebook, sparse_mask


def decompress_f(compressed_bits: str, huffman_codebook: dict, n_values: int) -> np.ndarray:
    reverse_codebook = {v: k for k, v in huffman_codebook.items()}

    result = []
    buffer = ""
    for bit in compressed_bits:
        buffer += bit
        if buffer in reverse_codebook:
            result.append(reverse_codebook[buffer])
            buffer = ""
        if len(result) == n_values:
            break

    return np.array(result, dtype=np.int32)
