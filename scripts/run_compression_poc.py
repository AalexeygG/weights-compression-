import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from s1p2_io import unpack_bytes_to_nibbles, decode_s1p2
from compression import compress_f, decompress_f
from metrics import compression_ratio, compute_mse, compute_hamming_distance


def run_on_file(path: str, sparsity_levels=(10, 20, 30, 40)):
    path = Path(path)
    raw_bytes = path.read_bytes()
    nibbles = unpack_bytes_to_nibbles(raw_bytes)
    n_values = len(nibbles)
    original_decoded = decode_s1p2(nibbles)

    print(f"file: {path.name}")
    print(f"values: {n_values:,}\n")

    for sparsity in sparsity_levels:
        t0 = time.perf_counter()
        compressed_bits, codebook, sparse_mask = compress_f(nibbles, sparsity)
        t1 = time.perf_counter()

        restored_nibbles = decompress_f(compressed_bits, codebook, n_values)
        t2 = time.perf_counter()

        restored_decoded = decode_s1p2(restored_nibbles)

        ratio = compression_ratio(n_values * 4, len(compressed_bits))
        mse, rmse = compute_mse(original_decoded, restored_decoded)
        hamming_bits, hamming_pct = compute_hamming_distance(nibbles, restored_nibbles)

        print(f"--- sparsity target: {sparsity}% ---")
        print(f"  compression ratio: {ratio:.3f}x")
        print(f"  actual sparsity:   {sparse_mask.mean()*100:.1f}%")
        print(f"  mse / rmse:        {mse:.6f} / {rmse:.6f}")
        print(f"  hamming distance:  {hamming_bits} bits ({hamming_pct:.2f}%)")
        print(f"  compress time:     {(t1-t0)*1000:.1f} ms")
        print(f"  decompress time:   {(t2-t1)*1000:.1f} ms")
        print()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python scripts/run_compression_poc.py <path_to_file.bin>")
        sys.exit(1)
    run_on_file(sys.argv[1])
