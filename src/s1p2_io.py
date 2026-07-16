import numpy as np
from pathlib import Path


def decode_s1p2(nibble: np.ndarray) -> np.ndarray:
    nibble = nibble.astype(np.int32)
    sign = (nibble >> 3) & 1
    integer_bit = (nibble >> 2) & 1
    frac_bits = nibble & 0b11
    magnitude = integer_bit * 1.0 + frac_bits * 0.25
    return np.where(sign == 1, -magnitude, magnitude).astype(np.float32)


def unpack_bytes_to_nibbles(raw_bytes: bytes, low_first: bool = True) -> np.ndarray:
    data = np.frombuffer(raw_bytes, dtype=np.uint8)
    low = (data & 0x0F).astype(np.int32)
    high = ((data >> 4) & 0x0F).astype(np.int32)

    result = np.empty(len(data) * 2, dtype=np.int32)
    if low_first:
        result[0::2] = low
        result[1::2] = high
    else:
        result[0::2] = high
        result[1::2] = low
    return result


def load_s1p2_file(path: str | Path, shape: tuple[int, int] | None = None) -> np.ndarray:
    path = Path(path)
    raw_bytes = path.read_bytes()
    nibbles = unpack_bytes_to_nibbles(raw_bytes)
    decoded = decode_s1p2(nibbles)

    if shape is not None:
        expected = shape[0] * shape[1]
        if expected != len(decoded):
            raise ValueError(f"shape {shape} ({expected} elems) doesn't match decoded length {len(decoded)}")
        decoded = decoded.reshape(shape)

    return decoded


def infer_matrix_shape(n_elements: int, hidden_size: int = 4096) -> tuple[int, int] | None:
    if n_elements % hidden_size != 0:
        return None
    return (hidden_size, n_elements // hidden_size)
