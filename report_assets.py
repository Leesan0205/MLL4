import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pandas as pd

from common import RESULTS

FIG = RESULTS / "figures"


def save_chart(path):
    plt.tight_layout()
    plt.savefig(FIG / path, dpi=190, bbox_inches="tight")
    plt.close()


def main():
    if not (RESULTS / "rf_sweeps.csv").exists():
        raise FileNotFoundError("Missing results/rf_sweeps.csv; run Phase 3 first.")
    FIG.mkdir(parents=True, exist_ok=True)
    sweeps = pd.read_csv(RESULTS / "rf_sweeps.csv", keep_default_na=False)

    for feature, g in sweeps.groupby("parameter", sort=False):
        plt.figure(figsize=(6.6, 4.0))
        xs = list(range(len(g)))
        plt.errorbar(xs, g["macro_f1_mean"].astype(float).to_numpy(),
                     yerr=g["macro_f1_std"].astype(float).to_numpy(), marker="o", capsize=3)
        plt.xticks(xs, g["value"].astype(str).tolist())
        plt.xlabel(feature)
        plt.ylabel("Development CV macro-F1 (mean ± fold SD)")
        plt.title(f"Random Forest: effect of {feature}")
        plt.grid(alpha=0.2)
        save_chart(f"rf_sweep_{feature}.png")

    made = sorted(FIG.glob("rf_sweep_*.png"))
    print(f"Created {len(made)} sweep figures in {FIG}:")
    for png in made:
        print(f"  figures/{png.name}")


if __name__ == "__main__":
    main()