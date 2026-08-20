#!/usr/bin/env python3
"""Side-by-side persona-space scatter (one panel per model, each in its OWN PCA basis), in the same
style as the earlier interactive artifact, but built from the latest full runs (222/233 roles) with
the same preprocessing as the rest of the new analysis (per-dimension z-scoring before PCA).

Panels are fit independently; PC signs are matched across panels via shared-role loadings (PC1
oriented so the assistant end points right). The black arrow is each model's assistant axis
projected into its own PC1-PC2 plane, drawn from the role centroid; the star is the default persona.

Usage: python scripts/plot_side_by_side.py --data <snapshot dir> --strata /tmp/artifact_strata.json \
           --out exp/figures/persona_space_side_by_side.png [--layer 14]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

COLORS = {"assistant_adjacent": "#2a78d6", "neutral_human": "#1baf7a",
          "nonhuman_fantastical": "#eb6834", "other": "#9aa1b0"}
LABELS = {"assistant_adjacent": "assistant-adjacent", "neutral_human": "neutral human",
          "nonhuman_fantastical": "non-human / fantastical", "other": "other roles"}


def load_release(root, key):
    d = Path(root) / key / "release" / key
    axis = torch.load(d / "assistant_axis.pt", weights_only=False).float().numpy()
    default = torch.load(d / "default_vector.pt", weights_only=False).float().numpy()
    roles = {p.stem: torch.load(p, weights_only=False).float().numpy() for p in sorted((d / "role_vectors").glob("*.pt"))}
    return axis, default, roles


def panel_data(rel, layer):
    from sklearn.decomposition import PCA
    axis, default, roles = rel
    names = sorted(roles)
    X = np.stack([roles[n][layer] for n in names])
    mu, sd = X.mean(0), X.std(0) + 1e-8
    Z = (X - mu) / sd
    pca = PCA(n_components=2).fit(Z)
    P = Z @ pca.components_.T
    pdef = ((default[layer] - mu) / sd) @ pca.components_.T
    a = (axis[layer] / sd)                      # axis in z-units
    parr = a @ pca.components_.T
    cos1 = abs(float(a @ pca.components_[0] / (np.linalg.norm(a) + 1e-12)))
    # orient PC1 so the default (assistant end) is on the right
    if pdef[0] < np.median(P[:, 0]):
        P[:, 0] *= -1; pdef[0] *= -1; parr[0] *= -1; pca.components_[0] *= -1
    return {"names": names, "P": P, "pdef": pdef, "parr": parr,
            "var": pca.explained_variance_ratio_, "cos1": cos1, "load": pca.components_, "Z": Z}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--treatment", default="t0-mt-3b")
    ap.add_argument("--control", default="vanilla-3b")
    ap.add_argument("--strata", required=True)
    ap.add_argument("--layer", type=int, default=14)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    strata = json.load(open(args.strata))
    panels = []
    for key in (args.treatment, args.control):
        panels.append((key, panel_data(load_release(args.data, key), args.layer)))

    # match PC2 sign across panels via shared-role projections (PC1 already oriented by the default)
    ref = panels[0][1]
    other = panels[1][1]
    common = sorted(set(ref["names"]) & set(other["names"]))
    ridx = {n: i for i, n in enumerate(ref["names"])}; oidx = {n: i for i, n in enumerate(other["names"])}
    a2 = np.array([ref["P"][ridx[n], 1] for n in common]); b2 = np.array([other["P"][oidx[n], 1] for n in common])
    if np.corrcoef(a2, b2)[0, 1] < 0:
        other["P"][:, 1] *= -1; other["pdef"][1] *= -1; other["parr"][1] *= -1

    fig, axes = plt.subplots(1, 2, figsize=(17, 8.2))
    fig.patch.set_facecolor("#fcfcfb")
    for (key, pd), ax in zip(panels, axes):
        ax.set_facecolor("#fcfcfb")
        names, P = pd["names"], pd["P"]
        cats = [strata.get(n, "other") for n in names]
        for cat, col in COLORS.items():
            idx = [i for i, c in enumerate(cats) if c == cat]
            ax.scatter(P[idx, 0], P[idx, 1], s=26 if cat != "other" else 18, c=col,
                       alpha=0.9 if cat != "other" else 0.55, edgecolors="#fcfcfb", linewidths=0.5,
                       zorder=4 if cat != "other" else 3)
        # assistant-axis arrow from the centroid, scaled to ~1/3 of the plot span
        span = min(np.ptp(P[:, 0]), np.ptp(P[:, 1]))
        v = pd["parr"] / (np.linalg.norm(pd["parr"]) + 1e-12) * span * 0.33
        ax.annotate("", xy=(v[0], v[1]), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", color="#1c2230", lw=1.6), zorder=6)
        ax.annotate("assistant axis", (v[0], v[1]), xytext=(6, 4), textcoords="offset points",
                    fontsize=9, fontstyle="italic", color="#1c2230", zorder=6)
        ax.scatter([pd["pdef"][0]], [pd["pdef"][1]], s=210, marker="*", c="#1c2230",
                   edgecolors="#fcfcfb", linewidths=0.8, zorder=7)
        ax.annotate("default", (pd["pdef"][0], pd["pdef"][1]), xytext=(7, -11),
                    textcoords="offset points", fontsize=8.5, fontweight="bold", zorder=7)
        # labels: PC1/PC2 extremes + all assistant-adjacent + a few interest roles
        to_label = set(np.argsort(P[:, 0])[:9]) | set(np.argsort(P[:, 0])[-9:]) \
                 | set(np.argsort(P[:, 1])[:6]) | set(np.argsort(P[:, 1])[-6:]) \
                 | {i for i, c in enumerate(cats) if c == "assistant_adjacent"} \
                 | {i for i, n in enumerate(names) if n in ("robot", "tulpa", "pirate", "demon", "poet")}
        for i in to_label:
            ax.annotate(names[i], (P[i, 0], P[i, 1]), xytext=(4, 3), textcoords="offset points",
                        fontsize=6.4, color="#52514e", zorder=5)
        ax.axhline(0, color="#e3e6ec", lw=0.8, zorder=1); ax.axvline(0, color="#e3e6ec", lw=0.8, zorder=1)
        ax.set_xlabel(f"PC1 ({pd['var'][0]:.1%} var)   →  more assistant-like", fontsize=10)
        ax.set_ylabel(f"PC2 ({pd['var'][1]:.1%} var)", fontsize=10)
        ax.set_title(f"{key}   —   |cos(PC1, axis)| {pd['cos1']:.3f}, {len(names)} roles + default, layer {args.layer}",
                     fontsize=11, loc="left", fontfamily="monospace")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markersize=8, label=LABELS[k])
               for k, c in COLORS.items()]
    handles.append(Line2D([0], [0], marker="*", color="w", markerfacecolor="#1c2230", markersize=13, label="default persona"))
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Persona space per model (each panel its own PCA on z-scored role vectors; PC signs matched across panels)",
                 fontsize=12.5, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    fig.savefig(args.out, dpi=170, bbox_inches="tight", facecolor=fig.get_facecolor())
    print("wrote", args.out)
    # cross-panel loading correlations for the caption
    a1 = np.array([ref["P"][ridx[n], 0] for n in common]); b1 = np.array([other["P"][oidx[n], 0] for n in common])
    print(f"PC1 loading corr {abs(np.corrcoef(a1, b1)[0,1]):.3f}; PC2 loading corr {abs(np.corrcoef(a2, b2)[0,1]):.3f} over {len(common)} shared roles")


if __name__ == "__main__":
    main()
