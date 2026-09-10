"""
ml/eda.py
---------
Exploratory data analysis for water_potability.csv.

Produces:
  1. Missing-value summary
  2. Per-class feature statistics
  3. Correlation matrix (printed + saved as PNG)
  4. Feature distribution histograms by Potability class (saved as PNG)
  5. Boxplots per feature by class (saved as PNG)

Output images are written to  ml/plots/  (created automatically).

Run:
    python ml/eda.py
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend — no display required
import matplotlib.pyplot as plt
import seaborn as sns

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "water_potability.csv")
PLOT_DIR  = os.path.join(BASE_DIR, "ml", "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

TARGET   = "Potability"
FEATURES = [
    "ph", "Hardness", "Solids", "Chloramines",
    "Sulfate", "Conductivity", "Organic_carbon",
    "Trihalomethanes", "Turbidity",
]

sns.set_theme(style="whitegrid", palette="muted")


# ── Helpers ───────────────────────────────────────────────────────────────────
def _save(fig: plt.Figure, name: str) -> None:
    path = os.path.join(PLOT_DIR, name)
    fig.savefig(path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    print(f"  [saved]  {os.path.relpath(path)}")


# ── 1. Dataset overview ───────────────────────────────────────────────────────
def overview(df: pd.DataFrame) -> None:
    print("=" * 60)
    print("1. DATASET OVERVIEW")
    print("=" * 60)
    print(f"  Rows: {df.shape[0]}  |  Columns: {df.shape[1]}")
    print(f"  Features  : {FEATURES}")
    print(f"  Target    : {TARGET}")

    missing = df.isnull().sum()
    pct     = (missing / len(df) * 100).round(2)
    mv      = pd.DataFrame({"missing": missing, "%": pct})
    mv      = mv[mv["missing"] > 0]
    print("\n  Missing values:")
    if mv.empty:
        print("    None")
    else:
        print(mv.to_string(index=True))


# ── 2. Descriptive statistics ─────────────────────────────────────────────────
def descriptive_stats(df: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("2. DESCRIPTIVE STATISTICS  (all rows)")
    print("=" * 60)
    print(df[FEATURES].describe().round(4).to_string())


# ── 3. Per-class feature statistics ──────────────────────────────────────────
def per_class_stats(df: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("3. PER-CLASS FEATURE MEANS  (0 = not potable, 1 = potable)")
    print("=" * 60)
    grouped = df.groupby(TARGET)[FEATURES].mean().round(4)
    print(grouped.to_string())

    diff = grouped.loc[1] - grouped.loc[0]
    print("\n  Mean difference (potable - not potable):")
    print(diff.round(4).to_string())


# ── 4. Class distribution ─────────────────────────────────────────────────────
def class_distribution(df: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("4. CLASS DISTRIBUTION")
    print("=" * 60)
    vc  = df[TARGET].value_counts().sort_index()
    pct = (df[TARGET].value_counts(normalize=True).sort_index() * 100).round(2)
    for cls in vc.index:
        label = "Not potable" if cls == 0 else "Potable    "
        print(f"  {cls} ({label}): {vc[cls]:>5} rows  ({pct[cls]:.2f}%)")

    # Bar chart
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(["Not potable (0)", "Potable (1)"], vc.values,
           color=["#e07070", "#70a0e0"], edgecolor="white")
    ax.set_title("Class Distribution - Potability")
    ax.set_ylabel("Count")
    for i, v in enumerate(vc.values):
        ax.text(i, v + 15, str(v), ha="center", fontsize=11)
    _save(fig, "class_distribution.png")


# ── 5. Correlation matrix ─────────────────────────────────────────────────────
def correlation_matrix(df: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("5. CORRELATION MATRIX  (Pearson, features + target)")
    print("=" * 60)
    corr = df[FEATURES + [TARGET]].corr().round(3)
    print(corr.to_string())

    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.zeros_like(corr, dtype=bool)
    mask[np.triu_indices_from(mask)] = True
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
        center=0, linewidths=0.5, ax=ax,
        annot_kws={"size": 9},
    )
    ax.set_title("Feature Correlation Matrix")
    _save(fig, "correlation_matrix.png")


# ── 6. Feature distributions by class ────────────────────────────────────────
def feature_distributions(df: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("6. FEATURE DISTRIBUTIONS  (histograms by class)")
    print("=" * 60)
    fig, axes = plt.subplots(3, 3, figsize=(14, 11))
    axes = axes.flatten()

    for i, feat in enumerate(FEATURES):
        ax = axes[i]
        for cls, color, label in [(0, "#e07070", "Not potable"), (1, "#70a0e0", "Potable")]:
            subset = df[df[TARGET] == cls][feat].dropna()
            ax.hist(subset, bins=40, alpha=0.6, color=color, label=label, density=True)
        ax.set_title(feat, fontsize=10)
        ax.set_xlabel("")
        ax.set_ylabel("Density")
        if i == 0:
            ax.legend(fontsize=8)

    fig.suptitle("Feature Distributions by Potability Class", fontsize=13, y=1.01)
    fig.tight_layout()
    _save(fig, "feature_distributions.png")
    print("  (see ml/plots/feature_distributions.png)")


# ── 7. Boxplots by class ──────────────────────────────────────────────────────
def boxplots(df: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("7. BOXPLOTS BY CLASS")
    print("=" * 60)
    fig, axes = plt.subplots(3, 3, figsize=(14, 11))
    axes = axes.flatten()

    for i, feat in enumerate(FEATURES):
        ax = axes[i]
        data = [
            df[df[TARGET] == 0][feat].dropna().values,
            df[df[TARGET] == 1][feat].dropna().values,
        ]
        bp = ax.boxplot(data, patch_artist=True, widths=0.5,
                        medianprops={"color": "black", "linewidth": 2})
        bp["boxes"][0].set_facecolor("#e07070")
        bp["boxes"][1].set_facecolor("#70a0e0")
        ax.set_xticklabels(["Not potable", "Potable"], fontsize=8)
        ax.set_title(feat, fontsize=10)

    fig.suptitle("Feature Boxplots by Potability Class", fontsize=13, y=1.01)
    fig.tight_layout()
    _save(fig, "boxplots.png")
    print("  (see ml/plots/boxplots.png)")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"\nLoading  {DATA_PATH}\n")
    df = pd.read_csv(DATA_PATH)

    overview(df)
    descriptive_stats(df)
    per_class_stats(df)
    class_distribution(df)
    correlation_matrix(df)
    feature_distributions(df)
    boxplots(df)

    print("\n" + "=" * 60)
    print("EDA complete.")
    print(f"Plots saved to: {os.path.relpath(PLOT_DIR)}")
    print("=" * 60)
