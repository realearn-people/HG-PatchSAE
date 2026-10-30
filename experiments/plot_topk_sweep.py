"""
plot_topk_sweep.py
==================
Publication-quality line plot of the Top-K masking sweep across all
evaluated methods. Reads metrics.csv files from outputs/ff_masking
and writes the Fig. 3 sweep plot (pdf/svg/png).

Left panel  : ON_K accuracy vs K (top-K features kept on, rest zeroed).
Right panel : OFF_K accuracy vs K (top-K features zeroed, rest kept).

Usage:
    PYTHONPATH="." python experiments/plot_topk_sweep.py
    PYTHONPATH="." python experiments/plot_topk_sweep.py \\
        --results_root outputs/ff_masking/sae_base \\
        --layout lr --output outputs/masking_sweep.pdf   # Fig. 3
"""
import argparse
import glob
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# K values evaluated by feature_faithfulness.py (49152 = full SAE dim).
TOPK = [1, 2, 5, 10, 50, 100, 500, 1000, 2000, 49152]


def _xtick_label(k):
    """Human-friendly tick labels for the log axis."""
    if k == 49152:
        return "49,152"
    if k >= 1000:
        return f"{k // 1000}k"
    return str(k)

# Methods in the order of Table 1 (ascending FF_100): (folder_slug, display_label, style_kwargs).
METHODS = [
    ("uniform",                 "Uniform",
        dict(color="#888888", linestyle="--", marker="o", zorder=2)),
    ("random",                  "Random",
        dict(color="#555555", linestyle="--", marker="s", zorder=2)),
    ("raw_attention",           "Raw Attention",
        dict(color="#d62728", linestyle="-",  marker="x", zorder=3)),
    ("gradcam",                 "GradCAM",
        dict(color="#ff7f0e", linestyle="-",  marker="v", zorder=3)),
    ("attention_rollout",       "Attention Rollout",
        dict(color="#9467bd", linestyle="-",  marker="^", zorder=4)),
    ("legrad",                  "LeGrad",
        dict(color="#8c564b", linestyle="-",  marker="D", zorder=5)),
    ("transformer_attribution", "Transformer Attribution",
        dict(color="#17becf", linestyle="-",  marker="P", zorder=5)),
    ("grad_sam",                "Grad-SAM",
        dict(color="#2ca02c", linestyle="-",  marker="h", zorder=5)),
    ("layercam",                "LayerCAM",
        dict(color="#0d47a1", linestyle="-",  marker="*", zorder=6, markersize=8)),
]

ROW_LABELS = ["no_sae"]
for k in TOPK:    # metrics.csv rows: no_sae, then (on_K, off_K) for each K
    ROW_LABELS += [f"on_{k}", f"off_{k}"]


def load_method_means(csv_path):
    df = pd.read_csv(csv_path)
    return dict(zip(ROW_LABELS, df.mean(axis=1).values))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results_root", default="outputs/ff_masking/sae_base",
        help="Root containing <method>/*/imagenet/metrics.csv")
    p.add_argument("--layout", choices=["tb", "lr"], default="tb",
        help="Panel arrangement: tb = ON over OFF (top-bottom, large, default); "
             "lr = ON | OFF (left-right, the original).")
    p.add_argument("--exclude", nargs="*", default=[],
        help="Method display names to omit from the plot, e.g. --exclude Random")
    p.add_argument("--hide_vit", action="store_true",
        help="Omit the 'Unmodified ViT' reference line and label.")
    p.add_argument("--output", default=None,
        help="Output PDF path (.svg/.png saved alongside). "
             "Default: outputs/masking_sweep_tb.pdf (tb) or outputs/masking_sweep.pdf (lr).")
    args = p.parse_args()
    tb = args.layout == "tb"
    if args.output is None:
        args.output = "outputs/masking_sweep_tb.pdf" if tb else "outputs/masking_sweep.pdf"

    excl = set(args.exclude or [])
    methods = [e for e in METHODS if e[1] not in excl]

    # ── Load ───────────────────────────────────────────────────────────────
    data = {}
    for slug, label, _ in methods:
        matches = glob.glob(os.path.join(args.results_root, slug, "*", "imagenet",
                                          "metrics.csv"))
        if not matches:
            print(f"  [{slug}] no metrics.csv found, skipping")
            continue
        data[label] = load_method_means(matches[0])
    if not data:
        raise SystemExit(f"No metrics found under {args.results_root}")

    no_sae = float(np.mean([d["no_sae"] for d in data.values()]))

    # ── Style ──────────────────────────────────────────────────────────────
    fs = 1.3 if tb else 1.0  # scale fonts up for the larger stacked figure
    mpl.rcParams.update({
        "font.family":       "serif",
        "font.size":         10 * fs,
        "axes.titlesize":    11 * fs,
        "axes.labelsize":    10 * fs,
        "axes.titleweight":  "bold",
        "legend.fontsize":   8.5 * fs,
        "legend.frameon":    False,
        "xtick.labelsize":   9 * fs,
        "ytick.labelsize":   9 * fs,
        "savefig.dpi":       300,
        "savefig.bbox":      "tight",
        "axes.spines.top":   False,
        "axes.spines.right": False,
    })

    if tb:
        fig, (ax_on, ax_off) = plt.subplots(2, 1, figsize=(9.5, 11.5),
                                            constrained_layout=False)
    else:
        fig, (ax_on, ax_off) = plt.subplots(1, 2, figsize=(11, 4.2),
                                            constrained_layout=False)

    # ── Plot ───────────────────────────────────────────────────────────────
    for slug, label, style in methods:
        if label not in data:
            continue
        d   = data[label]
        on  = [d[f"on_{k}"]  for k in TOPK]
        off = [d[f"off_{k}"] for k in TOPK]
        defaults = dict(linewidth=1.6, markersize=5, markeredgewidth=0.6,
                        markeredgecolor="white")
        s = {**defaults, **style}
        ax_on.plot(TOPK,  on,  label=label, **s)
        ax_off.plot(TOPK, off, label=label, **s)

    # Unmodified ViT reference on both panels. The label gets an opaque white
    # box so the dashed reference line does not strike through the text.
    if not args.hide_vit:
        for ax in (ax_on, ax_off):
            ax.axhline(no_sae, color="black", linewidth=0.7, alpha=0.5,
                       linestyle=(0, (4, 3)), zorder=1)
            ax.text(TOPK[0] * 1.15, no_sae,
                    f"Unmodified ViT ({no_sae:.1f}%)",
                    ha="left", va="center", fontsize=8 * fs, alpha=0.85, zorder=7,
                    bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                              edgecolor="none"))

    # ── Axes ───────────────────────────────────────────────────────────────
    for ax in (ax_on, ax_off):
        ax.set_xscale("log")
        ax.set_xticks(TOPK)
        ax.set_xticklabels([_xtick_label(k) for k in TOPK], fontsize=8.5 * fs)
        ax.minorticks_off()
        ax.set_xlabel(r"$K$  (top-ranked SAE features)")
        ax.grid(True, which="major", linestyle=":", linewidth=0.45, alpha=0.55)
        ax.set_xlim(TOPK[0] * 0.85, TOPK[-1] * 1.6)

    ax_on.set_title(r"Top-$K$ ON")
    ax_on.set_ylabel(r"$\mathrm{ON}_K$ accuracy (%)")

    ax_off.set_title(r"Top-$K$ OFF")
    ax_off.set_ylabel(r"$\mathrm{OFF}_K$ accuracy (%)")

    # ── Shared legend below ────────────────────────────────────────────────
    handles, labels = ax_on.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center",
               ncol=3 if tb else 5, bbox_to_anchor=(0.5, -0.01 if tb else -0.03),
               handlelength=2.2, columnspacing=1.4, borderaxespad=0.0)

    fig.tight_layout(rect=[0, 0.11 if tb else 0.10, 1, 1.0])

    out_pdf = args.output
    out_svg = os.path.splitext(out_pdf)[0] + ".svg"
    out_png = os.path.splitext(out_pdf)[0] + ".png"
    os.makedirs(os.path.dirname(out_pdf) or ".", exist_ok=True)
    fig.savefig(out_pdf, format="pdf", bbox_inches="tight")
    fig.savefig(out_svg, format="svg", bbox_inches="tight")
    fig.savefig(out_png, format="png", bbox_inches="tight")
    print(f"Saved: {out_pdf}")
    print(f"Saved: {out_svg}")
    print(f"Saved: {out_png}")


if __name__ == "__main__":
    main()
