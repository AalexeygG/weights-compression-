import numpy as np


def compression_ratio(original_bits: int, compressed_bits: int) -> float:
    return original_bits / compressed_bits


def compute_mse(original: np.ndarray, restored: np.ndarray) -> tuple[float, float]:
    mse = float(np.mean((original.astype(np.float64) - restored.astype(np.float64)) ** 2))
    rmse = float(np.sqrt(mse))
    return mse, rmse


def compute_hamming_distance(original_nibbles: np.ndarray, restored_nibbles: np.ndarray) -> tuple[int, float]:
    def to_bits(idx_array):
        return np.unpackbits(idx_array.astype(np.uint8).reshape(-1, 1), axis=1)[:, -4:]

    original_bits = to_bits(original_nibbles)
    restored_bits = to_bits(restored_nibbles)
    n_diff = int(np.sum(original_bits != restored_bits))
    pct_diff = n_diff / (len(original_nibbles) * 4) * 100
    return n_diff, pct_diff
