import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from s1p2_io import decode_s1p2, unpack_bytes_to_nibbles


def test_decode_known_values():
    nibbles = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
    expected = np.array([0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75,
                          -0.0, -0.25, -0.5, -0.75, -1.0, -1.25, -1.5, -1.75])
    decoded = decode_s1p2(nibbles)
    np.testing.assert_allclose(decoded, expected, atol=1e-6)


def test_symmetry():
    positive_nibbles = np.array([0, 1, 2, 3, 4, 5, 6, 7])
    negative_nibbles = np.array([8, 9, 10, 11, 12, 13, 14, 15])
    pos_decoded = decode_s1p2(positive_nibbles)
    neg_decoded = decode_s1p2(negative_nibbles)
    np.testing.assert_allclose(pos_decoded, -neg_decoded, atol=1e-6)


def test_unpack_bytes_roundtrip():
    # 0xA3 = 0b10100011 -> low nibble 0x3 = 3, high nibble 0xA = 10
    raw = bytes([0xA3])
    nibbles = unpack_bytes_to_nibbles(raw, low_first=True)
    assert list(nibbles) == [3, 10]

    nibbles_high_first = unpack_bytes_to_nibbles(raw, low_first=False)
    assert list(nibbles_high_first) == [10, 3]


def test_range_bounds():
    all_nibbles = np.arange(16)
    decoded = decode_s1p2(all_nibbles)
    assert decoded.min() >= -1.75
    assert decoded.max() <= 1.75


if __name__ == "__main__":
    test_decode_known_values()
    test_symmetry()
    test_unpack_bytes_roundtrip()
    test_range_bounds()
    print("all tests passed")
