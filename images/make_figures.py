"""Generate the report figures from already-saved OOF predictions / caches.
No training or GPU inference needed except the oracle-gap sweep (cheap: a
cosine top-k over cached descriptors).

Run: python images/make_figures.py

The architecture diagram (images/geomatch.png) is drawn by hand, not here.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "artifacts/oof_evidence"
TRAIN_IMAGES = Path("/var/tmp/luli38se-geomatch/data/geo_dataset/train")
FIG_DIR = Path(__file__).resolve().parent / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

ISO = ["BY", "DE", "ES", "FI", "FR", "GB", "IS", "IT", "NO", "PL", "SE", "TR"]
COUNTRY_NAME = {
    "BY": "Belarus",
    "DE": "Germany",
    "ES": "Spain",
    "FI": "Finland",
    "FR": "France",
    "GB": "United Kingdom",
    "IS": "Iceland",
    "IT": "Italy",
    "NO": "Norway",
    "PL": "Poland",
    "SE": "Sweden",
    "TR": "Turkey",
}
EARTH_KM = 6371.0088


def hav(a, b):
    a = np.radians(a)
    b = np.radians(b)
    dlat = b[..., 0] - a[..., 0]
    dlon = b[..., 1] - a[..., 1]
    h = (
        np.sin(dlat / 2) ** 2
        + np.cos(a[..., 0]) * np.cos(b[..., 0]) * np.sin(dlon / 2) ** 2
    )
    return 2 * EARTH_KM * np.arcsin(np.sqrt(np.clip(h, 0, 1)))


def load_baseline() -> pd.DataFrame:
    df = pd.read_csv(EVIDENCE / "predictions/baseline_pooled.csv")
    df["true_iso"] = df["true_country"].map(lambda i: ISO[i])
    return df


def load_final() -> pd.DataFrame:
    parts = []
    for f in range(5):
        parts.append(
            pd.read_csv(EVIDENCE / f"predictions/shipped_seed220517_fold{f}.csv")
        )
    df = pd.concat(parts, ignore_index=True)
    df["true_iso"] = df["true_country"].map(lambda i: ISO[i])
    return df


# ---------------------------------------------------------------- Figure A
def fig_per_country(baseline: pd.DataFrame, final: pd.DataFrame) -> None:
    m_baseline = baseline.groupby("true_iso")["distance_km"].median()
    m_final = final.groupby("true_iso")["distance_km"].median()
    order = m_final.sort_values().index
    x = np.arange(len(order))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    ax.bar(
        x - width / 2,
        m_baseline[order],
        width,
        label="Baseline (89.16 km pooled)",
        color="#c0392b",
    )
    ax.bar(
        x + width / 2,
        m_final[order],
        width,
        label="Final recipe (55.60 km pooled)",
        color="#2471a3",
    )
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(order)
    ax.set_ylabel("median error, km (log scale)")
    ax.set_title("Median error by country: baseline vs. final single model")
    ax.axhline(50, ls="--", c="gray", lw=0.9, zorder=0)
    ax.text(len(order) - 0.5, 55, "50 km", color="gray", fontsize=8, ha="right")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "per_country_median.png", dpi=180)
    plt.close(fig)
    print("wrote per_country_median.png")


# ---------------------------------------------------------------- Figure B
def fig_error_map(final: pd.DataFrame, n: int = 220) -> None:
    """Only draw lines for misses (>50 km); colour by whether the true
    country is DE/FR to show the errors are concentrated there, not spread
    evenly. Correct rows (<=50 km) are dots only -- a line would be invisible
    at this scale anyway."""
    misses = final[final.distance_km > 50].sample(n, random_state=0)
    correct = final[final.distance_km <= 50].sample(
        min(150, (final.distance_km <= 50).sum()), random_state=0
    )

    fig, ax = plt.subplots(figsize=(6.8, 6.8))
    is_defr = misses.true_iso.isin(["DE", "FR"])
    for _, r in misses[~is_defr].iterrows():
        ax.plot(
            [r.true_lng, r.predicted_lng],
            [r.true_lat, r.predicted_lat],
            color="#f0b27a",
            alpha=0.45,
            lw=0.7,
            zorder=1,
        )
    for _, r in misses[is_defr].iterrows():
        ax.plot(
            [r.true_lng, r.predicted_lng],
            [r.true_lat, r.predicted_lat],
            color="#c0392b",
            alpha=0.7,
            lw=1.1,
            zorder=2,
        )
    ax.scatter(
        correct.true_lng,
        correct.true_lat,
        s=8,
        c="#27ae60",
        zorder=3,
        label="correct (<=50 km)",
    )
    ax.scatter(
        misses[~is_defr].true_lng,
        misses[~is_defr].true_lat,
        s=8,
        c="#e67e22",
        zorder=2,
        label="miss, other country",
    )
    ax.scatter(
        misses[is_defr].true_lng,
        misses[is_defr].true_lat,
        s=10,
        c="#c0392b",
        zorder=3,
        label="miss, DE/FR",
    )
    ax.set_xlim(-25, 48)
    ax.set_ylim(33, 72)
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.set_aspect("equal")
    ax.set_title("Final model: misses are concentrated in Germany/France")
    ax.legend(loc="lower left", frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "error_map.png", dpi=180)
    plt.close(fig)
    print("wrote error_map.png")


# ---------------------------------------------------------------- Figure C
def error_decomposition() -> pd.DataFrame:
    """Split every query into: decoded within 50 km / lost to ranking / lost to
    recall. 'Lost to recall' means no <=50 km bank image was even retrieved into
    the top-200, so no reranker could have fixed it.

    Read from the committed evidence table rather than recomputed from the
    93 MB-per-fold descriptor caches, which are training outputs and are not in
    this repository; artifacts/oof_evidence/error_decomposition.json records
    which caches it was derived from and their SHA-256s.
    """
    table = pd.read_csv(EVIDENCE / "error_decomposition.csv")
    return pd.DataFrame(
        {
            "recalled": table["shortlist_min_km"].to_numpy() <= 50,
            "decoded": table["distance_km"].to_numpy() <= 50,
            "iso": table["iso"],
        }
    )


def fig_oracle_gap() -> None:
    frame = error_decomposition()
    rows = []
    for iso, part in frame.groupby("iso"):
        rows.append(
            {
                "iso": iso,
                "decoded": part.decoded.mean() * 100,
                "ranking": (part.recalled & ~part.decoded).mean() * 100,
                "recall": (~part.recalled).mean() * 100,
            }
        )
    table = pd.DataFrame(rows).sort_values("decoded", ascending=False)
    pooled = {
        "decoded": frame.decoded.mean() * 100,
        "ranking": (frame.recalled & ~frame.decoded).mean() * 100,
        "recall": (~frame.recalled).mean() * 100,
    }

    fig, ax = plt.subplots(figsize=(7.4, 3.2))
    x = np.arange(len(table))
    ax.bar(x, table.decoded, 0.72, color="#27ae60", label="located within 50 km")
    ax.bar(
        x,
        table.ranking,
        0.72,
        bottom=table.decoded,
        color="#e67e22",
        label="retrieved but mis-ranked",
    )
    ax.bar(
        x,
        table.recall,
        0.72,
        bottom=table.decoded + table.ranking,
        color="#c0392b",
        label="never retrieved (top-200)",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(table.iso)
    ax.set_ylim(0, 100)
    ax.set_ylabel("% of queries")
    ax.set_title(
        f"Pooled: {pooled['decoded']:.0f}% located  |  "
        f"{pooled['ranking']:.0f}% retrieved but mis-ranked  |  "
        f"{pooled['recall']:.0f}% never retrieved",
        fontsize=10,
    )
    ax.legend(
        frameon=False,
        fontsize=8.5,
        ncol=3,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.22),
    )
    fig.tight_layout()
    fig.savefig(FIG_DIR / "oracle_gap.png", dpi=180)
    plt.close(fig)
    print("wrote oracle_gap.png", {k: round(v, 1) for k, v in pooled.items()})


# ---------------------------------------------------------------- Figure D
# ---------------------------------------------------------------- Figure E
def load_curve(path: Path):
    xs, ys = [], []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        v = d.get("validation", {}).get("retrieval_top1", {})
        if "median_km" in v:
            xs.append(d["epoch"])
            ys.append(v["median_km"])
    return xs, ys


def _smooth(ys, window=3):
    if len(ys) < window:
        return ys
    kernel = np.ones(window) / window
    pad = window // 2
    padded = np.pad(np.asarray(ys, dtype=float), (pad, pad), mode="edge")
    return np.convolve(padded, kernel, mode="valid")[: len(ys)]


def fig_training_curves() -> None:
    x_variant, y_variant = load_curve(
        EVIDENCE / "curves/finer_tokens_seed900001_fold0.jsonl"
    )
    x_final, y_final = load_curve(EVIDENCE / "curves/shipped_seed220517_fold0.jsonl")
    y_variant_s, y_final_s = _smooth(y_variant), _smooth(y_final)
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ax.plot(x_variant, y_variant, color="#c0392b", alpha=0.25, lw=1)
    ax.plot(x_final, y_final, color="#2471a3", alpha=0.25, lw=1)
    ax.plot(
        x_variant,
        y_variant_s,
        color="#c0392b",
        lw=2,
        label="Longer-training variant: 8x8 tokens, 48 ep -- late-epoch drift up",
    )
    ax.plot(
        x_final,
        y_final_s,
        color="#2471a3",
        lw=2,
        label="Final recipe: 4x4 tokens, 40 ep",
    )
    best_i = int(np.argmin(y_variant_s))
    ax.scatter([x_variant[best_i]], [y_variant_s[best_i]], color="#c0392b", zorder=3)
    ax.annotate(
        f"best (smoothed) ep {x_variant[best_i]}\n{y_variant_s[best_i]:.1f} km, then drifts up\n"
        f"to {y_variant[-1]:.1f} km by ep {x_variant[-1]}",
        (x_variant[best_i], y_variant_s[best_i]),
        textcoords="offset points",
        xytext=(15, 25),
        fontsize=7.5,
        color="#c0392b",
        arrowprops=dict(arrowstyle="->", color="#c0392b", lw=0.8),
    )
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation median error, km (fold 0, 3-epoch smoothed)")
    ax.set_title("Validation curve: a failed retrain vs. the final recipe")
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "training_curves.png", dpi=180)
    plt.close(fig)
    print("wrote training_curves.png")


# ---------------------------------------------------------------- Figure F
def fig_examples(final: pd.DataFrame) -> None:
    easy = final[(final.true_iso.isin(["IS", "NO", "FI"])) & (final.distance_km < 8)]
    hard = final[(final.true_iso.isin(["DE", "FR"])) & (final.distance_km > 400)]
    easy_rows = easy.sample(2, random_state=1).to_dict("records")
    hard_rows = hard.sample(2, random_state=1).to_dict("records")

    fig, axes = plt.subplots(1, 4, figsize=(11, 3.3))
    rows = [easy_rows[0], easy_rows[1], hard_rows[0], hard_rows[1]]
    tags = ["correct", "correct", "wrong", "wrong"]
    colors = ["#27ae60", "#27ae60", "#c0392b", "#c0392b"]
    for ax, row, tag, color in zip(axes, rows, tags, colors):
        img = Image.open(TRAIN_IMAGES / row["filename"]).convert("RGB")
        ax.imshow(img)
        ax.axis("off")
        country = COUNTRY_NAME[ISO[row["true_country"]]]
        ax.set_title(
            f"{country}\n{tag}: {row['distance_km']:.0f} km off",
            fontsize=9,
            color=color,
        )
    fig.suptitle(
        "Easy (Nordic/Iceland, distinctive terrain) vs. hard (DE/FR, generic streets)",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(FIG_DIR / "examples.png", dpi=180)
    plt.close(fig)
    print("wrote examples.png")


REQUIRED_INPUTS = {
    "committed out-of-fold evidence": [
        EVIDENCE / "predictions/baseline_pooled.csv",
        *(EVIDENCE / f"predictions/shipped_seed220517_fold{f}.csv" for f in range(5)),
        EVIDENCE / "error_decomposition.csv",
        EVIDENCE / "curves/shipped_seed220517_fold0.jsonl",
        EVIDENCE / "curves/finer_tokens_seed900001_fold0.jsonl",
    ],
    "training images (for the example panel only)": [TRAIN_IMAGES],
}


def check_inputs() -> None:
    """These are cross-validation outputs, not repository files. A clean
    checkout has none of them; see the figure section of the README for which
    run produces each one."""
    missing = [
        f"  {path}   ({role})"
        for role, paths in REQUIRED_INPUTS.items()
        for path in paths
        if not path.exists()
    ]
    if missing:
        raise SystemExit(
            "Cannot draw the figures: these inputs are missing.\n"
            + "\n".join(missing)
            + "\n\nThey are produced by the cross-validation runs in the README; "
            "the committed figures under images/figures/ are the output of that "
            "same script on a machine that had them."
        )


def main() -> int:
    check_inputs()
    baseline = load_baseline()
    final = load_final()
    fig_per_country(baseline, final)
    fig_error_map(final)
    fig_oracle_gap()
    fig_training_curves()
    fig_examples(final)
    print(f"\nall figures in {FIG_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
