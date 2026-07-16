import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from eda import compare_files, pick_representative_sample
from visualize import plot_comparison

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "s1p2"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true", help="run over every file instead of the representative sample")
    args = parser.parse_args()

    bin_files = sorted(DATA_DIR.glob("*.bin"))
    if not bin_files:
        print(f"no .bin files found in {DATA_DIR}")
        return

    if args.all:
        paths = {p.stem: p for p in bin_files}
    else:
        paths = pick_representative_sample(bin_files)
        print(f"{len(bin_files)} files total, using representative sample of {len(paths)} (run with --all for everything)\n")

    stats = compare_files(paths)

    groups = {
        proj: sorted([n for n in stats if n.endswith(proj)])
        for proj in ["self_attn.q_proj", "self_attn.v_proj", "mlp.gate_proj", "mlp.down_proj"]
    }
    groups = {k: v for k, v in groups.items() if v}

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plot_comparison(stats, groups, save_path=FIGURES_DIR / "comparison.png")
    print(f"\nsaved: {FIGURES_DIR / 'comparison.png'}")


if __name__ == "__main__":
    main()
