"""
ff_significance.py   (any env with numpy + pandas + scipy; no GPU)
==================================================================
Error bars and significance tests for the Feature Faithfulness results,
computed from the metrics.csv files the ONE existing run already wrote.
No re-run is required -- and a re-run would not help, because the FF
pipeline is deterministic (frozen ViT + frozen SAE, prototypes built from
c2i[c][:130], a fixed val set, and a seeded random baseline). Running it
again reproduces the same numbers, so a "std over runs" would be 0 and
would say nothing about whether the effect is real.

The variability that actually matters is over the EVALUATION SAMPLE, and
that is already on disk: each metrics.csv row holds 1,000 per-class
accuracies, each from 50 validation images.

Three statistics, in increasing order of strength:

  1. sd     per-class dispersion (what topk_auc_analysis.py prints).
            DO NOT use this as an error bar -- it describes how much FF
            varies BETWEEN classes (some classes are just hard), not how
            precisely the mean is measured. It is large (~30) by nature.
  2. sem/CI standard error of the mean = sd/sqrt(1000), and the 95% CI.
            This is the "how precisely do we know this number" bar.
  3. paired Delta-FF vs Uniform computed PER CLASS, then averaged. Every
            method is evaluated on the same 1,000 classes and the same 50
            images per class, so pairing cancels class difficulty and is
            far tighter than comparing two independent CIs. This is the
            statistic that answers "is my method better than uniform".

Reported for FF_AUC and for FF at each individual K.

Outputs:
    outputs/ff_significance.json
    ff_significance_table.tex        (repo root, paper-ready)

Usage (from repo root):
    python experiments/ff_significance.py
    python experiments/ff_significance.py --k 100
"""

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
from scipy import stats

TOPK = [1, 2, 5, 10, 50, 100, 500, 1000, 2000, 49152]
ROW_LABELS = ["no_sae"]
for _k in TOPK:
    ROW_LABELS += [f"on_{_k}", f"off_{_k}"]

METHODS = [
    ("uniform", "Uniform"), ("random", "Random"),
    ("raw_attention", "Raw Attention"), ("gradcam", "GradCAM"),
    ("attention_rollout", "Attention Rollout"), ("legrad", "LeGrad"),
    ("transformer_attribution", "Transformer Attribution"),
    ("grad_sam", "Grad-SAM"), ("layercam", "LayerCAM"),
]

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--results_root", default="outputs/ff_masking/sae_base")
parser.add_argument("--baseline", default="uniform",
    help="Method the paired comparison is made against.")
parser.add_argument("--k", type=int, default=100,
    help="Single K reported in the LaTeX table alongside FF_AUC.")
parser.add_argument("--n_boot", type=int, default=10000,
    help="Bootstrap resamples over classes for the paired CI.")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--out_json", default="outputs/ff_significance.json")
parser.add_argument("--out_tex", default="outputs/ff_significance_table.tex")
args = parser.parse_args()
rng = np.random.default_rng(args.seed)


def load(slug):
    pat = os.path.join(args.results_root, slug, "*", "imagenet", "metrics.csv")
    m = glob.glob(pat)
    if not m:
        return None
    arr = pd.read_csv(m[0]).values.astype(np.float64)
    if arr.shape[0] < len(ROW_LABELS):
        raise ValueError(f"{m[0]}: {arr.shape[0]} rows, expected >= {len(ROW_LABELS)}")
    return {lbl: arr[i] for i, lbl in enumerate(ROW_LABELS)}


logk = np.log10(TOPK)
span = logk[-1] - logk[0]

data, ff_auc, ff_at_k = {}, {}, {}
for slug, label in METHODS:
    d = load(slug)
    if d is None:
        print(f"  [{slug}] no metrics.csv, skipping")
        continue
    data[slug] = d
    on = np.stack([d[f"on_{k}"] for k in TOPK])          # (10, 1000)
    off = np.stack([d[f"off_{k}"] for k in TOPK])
    ff_auc[slug] = (np.trapz(on, logk, axis=0) - np.trapz(off, logk, axis=0)) / span
    ff_at_k[slug] = {k: d[f"on_{k}"] - d[f"off_{k}"] for k in TOPK}

n_cls = len(next(iter(ff_auc.values())))
print(f"Loaded {len(ff_auc)} methods, {n_cls} classes, 50 val images per class.\n")


def describe(x):
    sd = x.std(ddof=1)
    sem = sd / np.sqrt(len(x))
    return {"mean": float(x.mean()), "sd": float(sd), "sem": float(sem),
            "ci95": [float(x.mean() - 1.96 * sem), float(x.mean() + 1.96 * sem)]}


def paired(x, base):
    d = x - base
    sd = d.std(ddof=1)
    sem = sd / np.sqrt(len(d))
    idx = rng.integers(0, len(d), size=(args.n_boot, len(d)))
    boot = d[idx].mean(1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    try:
        p = float(stats.wilcoxon(d)[1])
    except Exception:
        p = float("nan")
    # Accuracy is quantised to 2pp (50 images/class), so exact ties are common
    # at a single K -- counting them as losses deflates the win rate badly
    # (e.g. 67% ties for Random at K=100). Report the tie-excluded rate.
    wins, losses, ties = int((d > 0).sum()), int((d < 0).sum()), int((d == 0).sum())
    return {"delta": float(d.mean()), "sem": float(sem),
            "ci95_boot": [float(lo), float(hi)],
            "win_rate": float(wins / max(wins + losses, 1)),
            "win_rate_with_ties": float(wins / len(d)),
            "tie_rate": float(ties / len(d)), "wilcoxon_p": p,
            "cohens_d": float(d.mean() / sd) if sd > 0 else float("nan")}


BASE = args.baseline
lab = dict(METHODS)

print("=" * 100)
print(f"FF_AUC  --  mean, per-class sd, and the SE/CI of the mean (n = {n_cls} classes)")
print("=" * 100)
print(f"{'method':<26}{'FF_AUC':>9}{'sd':>8}{'sem':>7}{'95% CI of the mean':>24}")
summary = {}
for slug in ff_auc:
    s = describe(ff_auc[slug])
    summary[slug] = {"ff_auc": s}
    print(f"{lab[slug]:<26}{s['mean']:>9.2f}{s['sd']:>8.2f}{s['sem']:>7.2f}"
          f"   [{s['ci95'][0]:6.2f}, {s['ci95'][1]:6.2f}]")

print("\n" + "=" * 100)
print(f"PAIRED  Delta FF_AUC vs {lab[BASE]}  (same classes, same images -> pairing "
      f"cancels class difficulty)")
print("=" * 100)
print(f"{'method':<26}{'ΔFF_AUC':>9}{'95% CI (bootstrap)':>24}{'win*':>8}"
      f"{'ties':>7}{'Wilcoxon p':>13}{'d':>7}")
for slug in ff_auc:
    if slug == BASE:
        continue
    pr = paired(ff_auc[slug], ff_auc[BASE])
    summary[slug]["paired_ff_auc"] = pr
    star = "***" if pr["wilcoxon_p"] < 1e-3 else "**" if pr["wilcoxon_p"] < 1e-2 \
        else "*" if pr["wilcoxon_p"] < 0.05 else "ns"
    print(f"{lab[slug]:<26}{pr['delta']:>+9.2f}   "
          f"[{pr['ci95_boot'][0]:+6.2f}, {pr['ci95_boot'][1]:+6.2f}]"
          f"{pr['win_rate']:>8.1%}{pr['tie_rate']:>7.1%}"
          f"{pr['wilcoxon_p']:>11.1e} {star}{pr['cohens_d']:>7.2f}")

K = args.k
print("\n" + "=" * 100)
print(f"Same, at the single operating point K = {K}")
print("=" * 100)
_dhdr = "ΔFF vs " + lab[BASE]
print(f"{'method':<26}{'FF':>8}{'sem':>7}{_dhdr:>16}"
      f"{'95% CI':>22}{'win*':>8}{'ties':>7}{'Wilcoxon p':>13}")
for slug in ff_at_k:
    s = describe(ff_at_k[slug][K])
    summary.setdefault(slug, {})[f"ff_{K}"] = s
    if slug == BASE:
        print(f"{lab[slug]:<26}{s['mean']:>8.2f}{s['sem']:>7.2f}{'--':>16}")
        continue
    pr = paired(ff_at_k[slug][K], ff_at_k[BASE][K])
    summary[slug][f"paired_ff_{K}"] = pr
    star = "***" if pr["wilcoxon_p"] < 1e-3 else "**" if pr["wilcoxon_p"] < 1e-2 \
        else "*" if pr["wilcoxon_p"] < 0.05 else "ns"
    print(f"{lab[slug]:<26}{s['mean']:>8.2f}{s['sem']:>7.2f}{pr['delta']:>+16.2f}"
          f"   [{pr['ci95_boot'][0]:+6.2f}, {pr['ci95_boot'][1]:+6.2f}]"
          f"{pr['win_rate']:>8.1%}{pr['tie_rate']:>7.1%}"
          f"{pr['wilcoxon_p']:>11.1e} {star}")
print("=" * 100)

os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
with open(args.out_json, "w") as f:
    json.dump({"n_classes": int(n_cls), "imgs_per_class": 50, "baseline": BASE,
               "k_reported": K, "n_boot": args.n_boot, "methods": summary}, f, indent=2)

# ── Paper-ready LaTeX ────────────────────────────────────────────────────────
lines = [
    r"\begin{table}",
    r"\caption{Feature Faithfulness with uncertainty. The FF pipeline is",
    r"deterministic, so error bars are over the \emph{evaluation sample}: each",
    rf"value is a mean over {n_cls} classes ($50$ validation images per class).",
    r"$\mathrm{SEM}$ is the standard error of that mean. $\Delta\mathrm{FF}$ is",
    r"computed \emph{per class} against the uniform baseline and then averaged,",
    r"so it is paired; the CI is a $10{,}000$-sample bootstrap over classes and",
    r"$p$ is a two-sided Wilcoxon signed-rank test. `Win' is the fraction of",
    r"classes on which the method beats uniform.}",
    r"\label{tab:ff_significance}",
    r"\centering",
    r"\begin{tabular}{|l|c|c|c|c|}",
    r"\hline",
    r"\textbf{Method} & $\mathbf{FF_{AUC}}$ & $\boldsymbol{\Delta}\mathbf{FF_{AUC}}$ "
    r"& \textbf{95\% CI} & \textbf{Win} \\",
    r"\hline",
]
ordered = sorted([s for s in ff_auc], key=lambda s: summary[s]["ff_auc"]["mean"])
for slug in ordered:
    s = summary[slug]["ff_auc"]
    if slug == BASE:
        lines.append(rf"{lab[slug]} & ${s['mean']:.2f} \pm {s['sem']:.2f}$ "
                     rf"& --- & --- & --- \\")
    else:
        pr = summary[slug]["paired_ff_auc"]
        star = "***" if pr["wilcoxon_p"] < 1e-3 else "**" if pr["wilcoxon_p"] < 1e-2 \
            else "*" if pr["wilcoxon_p"] < 0.05 else r"\textsuperscript{ns}"
        lines.append(
            rf"{lab[slug]} & ${s['mean']:.2f} \pm {s['sem']:.2f}$ "
            rf"& ${pr['delta']:+.2f}$\textsuperscript{{{star}}} "
            rf"& $[{pr['ci95_boot'][0]:+.2f}, {pr['ci95_boot'][1]:+.2f}]$ "
            rf"& ${pr['win_rate']*100:.1f}\%$ \\")
lines += [r"\hline", r"\end{tabular}", r"\end{table}"]
with open(args.out_tex, "w") as f:
    f.write("\n".join(lines) + "\n")

print(f"\nSaved: {args.out_json}")
print(f"Saved: {args.out_tex}")
