"""
paper_tables.py
===============
Print the Feature Faithfulness columns of Table 1 and all of Table 2
(EPG vs FF_100, with the Pearson / Spearman correlation reported in Sec. 6).

Inputs (no GPU needed):
  * metrics.csv files written by feature_faithfulness.py
  * the JSON written by epg.py

Usage:
    python experiments/paper_tables.py \\
        --results_root outputs/ff_masking/sae_base \\
        --epg_json     outputs/epg_iou_results.json
"""
import argparse
import glob
import json
import os
import sys

from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_topk_sweep import METHODS, load_method_means  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--results_root", default="outputs/ff_masking/sae_base")
p.add_argument("--epg_json", default="outputs/epg_iou_results.json")
p.add_argument("--k", type=int, default=100)
args = p.parse_args()

# ── Table 1: Feature Faithfulness at K ────────────────────────────────────────
ff = {}
print(f"\nTable 1 — Feature Faithfulness (K = {args.k})")
print(f"{'Method':<25}{'ON':>8}{'OFF':>8}{'FF':>9}")
vit = None
for slug, label, _ in METHODS:
    matches = glob.glob(os.path.join(args.results_root, slug, "*", "imagenet", "metrics.csv"))
    if not matches:
        print(f"{label:<25}  (missing)")
        continue
    m = load_method_means(matches[0])
    vit = m["no_sae"]
    on, off = m[f"on_{args.k}"], m[f"off_{args.k}"]
    ff[label] = on - off
    print(f"{label:<25}{on:>8.2f}{off:>8.2f}{on - off:>+9.2f}")
if vit is not None:
    print(f"Unmodified ViT: {vit:.2f}%")

# ── Table 2: EPG vs FF and their correlation ──────────────────────────────────
if not os.path.exists(args.epg_json):
    print(f"\n{args.epg_json} not found — run experiments/epg.py for Table 2.")
    sys.exit(0)

with open(args.epg_json) as f:
    epg = json.load(f)

# Uniform is excluded: its EPG equals the mean bbox coverage, i.e. the
# same reference point as Random (Sec. 6.1).
rows = [(label, epg[label]["epg_mean"], ff[label])
        for _, label, _ in METHODS
        if label != "Uniform" and label in epg and label in ff]

print(f"\nTable 2 — EPG vs FF_{args.k}")
print(f"{'Method':<25}{'EPG':>8}{'FF':>9}")
for label, e, v in rows:
    print(f"{label:<25}{e:>8.3f}{v:>+9.2f}")
if "Uniform" in epg:
    print(f"(Uniform EPG = {epg['Uniform']['epg_mean']:.3f}, mean bbox coverage)")

if len(rows) >= 3:
    e = [r[1] for r in rows]
    v = [r[2] for r in rows]
    r, pr = pearsonr(e, v)
    rho, ps = spearmanr(e, v)
    print(f"\nn = {len(rows)} methods:  Pearson r = {r:.3f} (p = {pr:.4f}),  "
          f"Spearman rho = {rho:.3f} (p = {ps:.4f})")
