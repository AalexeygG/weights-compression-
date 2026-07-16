import re
import numpy as np
from pathlib import Path
from dataclasses import dataclass

from s1p2_io import unpack_bytes_to_nibbles, decode_s1p2

LAYER_RE = re.compile(r"layers\.(\d+)\.(.+?)\.weight")


@dataclass
class FileStats:
    name: str
    n_values: int
    mean: float
    std: float
    abs_mean: float
    zero_pct: float
    nibble8_count: int
    decoded: np.ndarray


def compute_file_stats(path: str | Path) -> FileStats:
    path = Path(path)
    raw_bytes = path.read_bytes()
    nibbles = unpack_bytes_to_nibbles(raw_bytes)
    decoded = decode_s1p2(nibbles)

    return FileStats(
        name=path.stem,
        n_values=len(decoded),
        mean=float(decoded.mean()),
        std=float(decoded.std()),
        abs_mean=float(np.abs(decoded).mean()),
        zero_pct=float(np.mean(decoded == 0) * 100),
        nibble8_count=int(np.sum(nibbles == 8)),
        decoded=decoded,
    )


def compare_files(paths: dict[str, str | Path]) -> dict[str, FileStats]:
    results = {name: compute_file_stats(p) for name, p in paths.items()}

    header = f"{'file':<45} {'n_values':>12} {'mean':>8} {'std':>8} {'mean|x|':>9} {'zero%':>8} {'nibble=8':>10}"
    print(header)
    print('-' * len(header))
    for name, r in results.items():
        print(f"{name:<45} {r.n_values:>12,} {r.mean:>8.4f} {r.std:>8.4f} "
              f"{r.abs_mean:>9.4f} {r.zero_pct:>7.2f}% {r.nibble8_count:>10}")

    zero_pcts = np.array([r.zero_pct for r in results.values()])
    median_zero = np.median(zero_pcts)
    for name, r in results.items():
        if abs(r.zero_pct - median_zero) > 10:
            print(f"\nANOMALY: {name} has {r.zero_pct:.1f}% zeros (median: {median_zero:.1f}%)")

    return results


def pick_representative_sample(bin_files: list[Path]) -> dict[str, Path]:
    by_proj = {}
    for p in bin_files:
        m = LAYER_RE.search(p.name)
        if not m:
            continue
        layer, proj = int(m.group(1)), m.group(2)
        by_proj.setdefault(proj, {})[layer] = p

    sample = {}
    for proj, layer_map in by_proj.items():
        layers = sorted(layer_map)
        first, last = layers[0], layers[-1]
        mid = layers[len(layers) // 2]
        for layer in {first, mid, last}:
            path = layer_map[layer]
            sample[f"L{layer}.{proj}"] = path

    return sample
