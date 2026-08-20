#!/usr/bin/env python3
"""SPP-vs-vanilla deep comparison: geometry preservation, persona movement, role-play behavior.

The two models are SEPARATE pretraining runs (unlike a LoRA fine-tune), so raw activation
directions are only partially aligned (per-role centered cos ~0.4 vs ~0.0 shuffled-role null).
Raw vector subtraction would therefore mostly measure coordinate-frame rotation between runs,
not persona movement. This script uses run-robust tools instead:

  1. RSA           — correlation of the two models' pairwise role-similarity matrices (alignment-free).
  2. PC loadings   — correlation of role loadings on each model's own PC1/2/3 (the assistant-axis
                     paper's cross-model yardstick; they report >0.92 on PC1 across model FAMILIES).
  3. Procrustes    — best orthogonal rotation of the vanilla role cloud onto the SPP cloud (both
                     z-scored), then per-role residual displacement = a LOWER BOUND on real movement.
  4. Judge scores  — per-role / per-category full-role-play (score-3) rates and score distributions.

Outputs figures to exp/figures/ and metrics to exp/spp_vs_vanilla_metrics.json.

Usage:  python scripts/spp_vs_vanilla_analysis.py --data <snapshot dir with t0-mt-3b/ vanilla-3b/> \
            --outdir exp [--layer 14]
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

CATEGORIES = {
    "assistant-like": """assistant tutor teacher instructor mentor coach guide counselor therapist translator
        interpreter editor proofreader summarizer secretary consultant facilitator mediator moderator
        librarian curator archivist grader evaluator examiner reviewer validator screener scheduler
        coordinator organizer planner dispatcher trainer navigator advocate caregiver healer synthesizer
        generalist specialist collaborator supervisor recruiter presenter""",
    "professional": """accountant analyst auditor economist statistician mathematician physicist chemist biologist
        scientist researcher scholar historian anthropologist archaeologist sociologist psychologist
        linguist geographer cartographer engineer architect builder mechanic programmer debugger
        technologist designer doctor paramedic pharmacist nutritionist veterinarian lawyer judge
        detective journalist reporter marketer merchant auctioneer entrepreneur producer publisher
        blogger podcaster photographer musician composer novelist writer playwright poet bard actor
        comedian chef bartender sommelier pilot soldier veteran ambassador emissary diplomat negotiator
        strategist forecaster futurist theorist philosopher critic interviewer influencer celebrity
        activist revolutionary politician""",
    "non-human": """demon angel ghost wraith revenant spirit vampire golem chimera leviathan whale predator
        prey parasite virus tree coral_reef mycorrhizal ecosystem wind void alien eldritch aberration
        familiar genie oracle hive swarm symbiont hybrid crystalline homunculus""",
    "AI / abstract": """robot cyborg simulacrum echo tulpa egregore avatar zeitgeist chameleon shapeshifter""",
}
ROLE2CAT = {r: c for c, roles in CATEGORIES.items() for r in roles.split()}
COLORS = {"assistant-like": "#2a78d6", "professional": "#eb6834", "human (other)": "#eda100",
          "non-human": "#1baf7a", "AI / abstract": "#4a3aa7"}
MALEVOLENT = ("demon wraith vampire parasite virus predator aberration eldritch criminal smuggler "
              "saboteur destroyer trickster rogue vigilante anarchist zealot narcissist").split()


def load_release(root, key):
    d = Path(root) / key / "release" / key
    axis = torch.load(d / "assistant_axis.pt", weights_only=False).float().numpy()
    default = torch.load(d / "default_vector.pt", weights_only=False).float().numpy()
    roles = {p.stem: torch.load(p, weights_only=False).float().numpy() for p in sorted((d / "role_vectors").glob("*.pt"))}
    return axis, default, roles


def load_scores(root, key):
    out = {}
    for f in sorted((Path(root) / key / "scores").glob("*.json")):
        if f.stem == "default":
            continue
        s = json.load(open(f))
        vals = [x for x in s.values() if isinstance(x, int)]
        out[f.stem] = [sum(1 for x in vals if x == k) for k in range(4)]
    return out


def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def zstd(X):
    return (X - X.mean(0)) / (X.std(0) + 1e-8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="dir containing t0-mt-3b/ and vanilla-3b/")
    ap.add_argument("--treatment", default="t0-mt-3b")
    ap.add_argument("--control", default="vanilla-3b")
    ap.add_argument("--layer", type=int, default=None)
    ap.add_argument("--outdir", default="exp")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from sklearn.decomposition import PCA

    t = load_release(args.data, args.treatment)   # treatment = SPP
    v = load_release(args.data, args.control)
    n_layers = t[0].shape[0]
    L = args.layer if args.layer is not None else n_layers // 2
    shared = sorted(set(t[2]) & set(v[2]))
    cats = [ROLE2CAT.get(n, "human (other)") for n in shared]
    figdir = Path(args.outdir) / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    M = {"layer": L, "n_shared": len(shared)}

    Xt = np.stack([t[2][n][L] for n in shared])
    Xv = np.stack([v[2][n][L] for n in shared])
    Xt_c, Xv_c = Xt - Xt.mean(0), Xv - Xv.mean(0)

    # ---- raw alignment level (context for why we don't subtract vectors) -------------------------
    pr = np.array([cos(Xt_c[i], Xv_c[i]) for i in range(len(shared))])
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(shared))
    M["raw_centered_cos_median"] = float(np.median(pr))
    M["raw_centered_cos_null"] = float(np.median([cos(Xt_c[i], Xv_c[perm[i]]) for i in range(len(shared))]))

    # ---- 1. RSA ---------------------------------------------------------------------------------
    def simmat(X):
        Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
        return Xn @ Xn.T
    iu = np.triu_indices(len(shared), 1)
    M["rsa_mid"] = float(np.corrcoef(simmat(Xt_c)[iu], simmat(Xv_c)[iu])[0, 1])
    M["rsa_per_layer"] = []
    for l in range(n_layers):
        Xt_l = np.stack([t[2][n][l] for n in shared]); Xv_l = np.stack([v[2][n][l] for n in shared])
        M["rsa_per_layer"].append([l, float(np.corrcoef(simmat(Xt_l - Xt_l.mean(0))[iu],
                                                        simmat(Xv_l - Xv_l.mean(0))[iu])[0, 1])])

    # ---- 2. PC-loading correlations -------------------------------------------------------------
    pt = PCA(n_components=5).fit(zstd(Xt)); pv = PCA(n_components=5).fit(zstd(Xv))
    Zt = zstd(Xt) @ pt.components_.T; Zv = zstd(Xv) @ pv.components_.T
    M["pc_loading_corr"] = [float(abs(np.corrcoef(Zt[:, k], Zv[:, k])[0, 1])) for k in range(3)]
    M["pc_var"] = {"spp": [float(x) for x in pt.explained_variance_ratio_[:3]],
                   "vanilla": [float(x) for x in pv.explained_variance_ratio_[:3]]}
    proj_t = Xt @ (t[0][L] / np.linalg.norm(t[0][L])); proj_v = Xv @ (v[0][L] / np.linalg.norm(v[0][L]))
    ra = np.argsort(np.argsort(proj_t)); rb = np.argsort(np.argsort(proj_v))
    M["axis_ordering_spearman"] = float(np.corrcoef(ra, rb)[0, 1])

    # ---- 3. Procrustes displacement -------------------------------------------------------------
    A, B = zstd(Xv), zstd(Xt)                      # map vanilla onto SPP
    U, s, Vt = np.linalg.svd(A.T @ B)
    R = U @ Vt
    A_rot = A @ R
    resid = np.linalg.norm(B - A_rot, axis=1)
    spread = np.linalg.norm(B - B.mean(0), axis=1)
    M["procrustes_disparity"] = float(np.sum((B - A_rot) ** 2) / np.sum(B ** 2))
    M["resid_over_spread_median"] = float(np.median(resid) / np.median(spread))
    dv_al = ((v[1][L] - Xv.mean(0)) / (Xv.std(0) + 1e-8)) @ R
    dt_z = (t[1][L] - Xt.mean(0)) / (Xt.std(0) + 1e-8)
    M["default_resid"] = float(np.linalg.norm(dt_z - dv_al))
    M["role_resid_median"] = float(np.median(resid))
    M["resid_by_role"] = {shared[i]: float(resid[i]) for i in range(len(shared))}

    # ---- 4. judge scores ------------------------------------------------------------------------
    st = load_scores(args.data, args.treatment); sv = load_scores(args.data, args.control)
    sroles = sorted(set(st) & set(sv))
    ht = np.sum([st[r] for r in sroles], axis=0); hv = np.sum([sv[r] for r in sroles], axis=0)
    M["score_dist"] = {"spp": (ht / ht.sum()).tolist(), "vanilla": (hv / hv.sum()).tolist()}
    r3t = {r: st[r][3] / sum(st[r]) for r in sroles}; r3v = {r: sv[r][3] / sum(sv[r]) for r in sroles}
    bycat = defaultdict(list)
    for r in sroles:
        bycat[ROLE2CAT.get(r, "human (other)")].append(r)
    bycat["malevolent (subset)"] = [r for r in MALEVOLENT if r in sroles]
    M["cat_score3"] = {c: {"vanilla": float(np.mean([r3v[r] for r in rs])),
                           "spp": float(np.mean([r3t[r] for r in rs])), "n": len(rs)} for c, rs in bycat.items()}

    # ================================ FIGURE 1: persona movement =================================
    # PCA basis fit on the ALIGNED union so both clouds live in one frame.
    pca = PCA(n_components=2).fit(np.vstack([A_rot, B]))
    Za = A_rot @ pca.components_.T   # vanilla, aligned
    Zb = B @ pca.components_.T       # SPP
    zda = dv_al @ pca.components_.T; zdb = dt_z @ pca.components_.T
    # orient PC1 so the SPP default points up
    if zdb[0] < np.median(Zb[:, 0]):
        Za[:, 0] *= -1; Zb[:, 0] *= -1; zda[0] *= -1; zdb[0] *= -1

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(16.5, 8), gridspec_kw={"width_ratios": [1.25, 1]})
    fig.patch.set_facecolor("#fcfcfb")
    for a in (ax, ax2):
        a.set_facecolor("#fcfcfb")
    for i in range(len(shared)):
        ax.annotate("", xy=(Zb[i, 1], Zb[i, 0]), xytext=(Za[i, 1], Za[i, 0]),
                    arrowprops=dict(arrowstyle="->", color="#9a9891", lw=0.6, alpha=0.7, shrinkA=0, shrinkB=0))
    for cat, col in COLORS.items():
        idx = [i for i, c in enumerate(cats) if c == cat]
        if not idx:
            continue
        ax.scatter(Za[idx, 1], Za[idx, 0], s=20, c=col, alpha=0.85, edgecolors="#fcfcfb", linewidths=0.5, zorder=3)
        ax.scatter(Zb[idx, 1], Zb[idx, 0], s=26, facecolors="none", edgecolors=col, linewidths=1.0, marker="D", zorder=3)
    ax.scatter([zda[1]], [zda[0]], s=200, marker="*", c="#4a3aa7", edgecolors="#0b0b0b", linewidths=0.8, zorder=6)
    ax.scatter([zdb[1]], [zdb[0]], s=200, marker="*", facecolors="none", edgecolors="#c0392b", linewidths=1.5, zorder=6)
    ax.annotate("default (vanilla)", (zda[1], zda[0]), xytext=(7, 5), textcoords="offset points", fontsize=8, fontweight="bold")
    ax.annotate("default (SPP)", (zdb[1], zdb[0]), xytext=(7, -11), textcoords="offset points", fontsize=8, fontweight="bold", color="#c0392b")
    order = np.argsort(resid)
    for i in list(order[-10:]) + list(order[:5]):
        ax.annotate(shared[i], (Zb[i, 1], Zb[i, 0]), xytext=(4, 3), textcoords="offset points", fontsize=6.5, color="#52514e")
    ax.set_xlabel("PC2 of the aligned union", fontsize=10)
    ax.set_ylabel("PC1 of the aligned union   →  more assistant-like", fontsize=10)
    ax.set_title(f"(a) Persona movement after optimal alignment (layer {L}) — ● vanilla → ◇ SPP\n"
                 f"Procrustes disparity {M['procrustes_disparity']:.2f}: only {M['procrustes_disparity']:.0%} of variance "
                 "is not explained by a rigid rotation between the two pretraining runs",
                 fontsize=10, loc="left")
    handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markersize=8, label=cat) for cat, c in COLORS.items()]
    ax.legend(handles=handles, loc="lower left", frameon=False, fontsize=8)

    # ranked residual displacement
    show = list(order[::-1][:22]) + [None] + list(order[:8][::-1])
    ys, labels, cols, vals = [], [], [], []
    y = 0
    for i in show:
        if i is None:
            y -= 1; continue
        ys.append(y); labels.append(shared[i]); cols.append(COLORS.get(cats[i], "#eda100")); vals.append(resid[i]); y -= 1
    ax2.barh(ys, vals, color=cols, height=0.8)
    ax2.set_yticks(ys); ax2.set_yticklabels(labels, fontsize=7.5)
    ax2.axvline(np.median(resid), color="#52514e", lw=1, ls="--")
    ax2.text(np.median(resid), ys[0] + 1.2, " median role", fontsize=8, color="#52514e")
    ax2.axvline(M["default_resid"], color="#c0392b", lw=1.2, ls=":")
    ax2.text(M["default_resid"], ys[-1] - 1.4, " default persona", fontsize=8, color="#c0392b")
    ax2.set_xlabel("residual displacement after alignment (z-scored units)", fontsize=9)
    ax2.set_title("(b) Most / least moved personas\n(top: most displaced — bottom block: most conserved)",
                  fontsize=10, loc="left")
    for a in (ax, ax2):
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
    fig.suptitle("Synthetic persona pretraining barely moves the persona landscape — and what moves is not the assistant",
                 fontsize=12.5, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(figdir / "spp_persona_movement.png", dpi=170, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

    # ================================ FIGURE 2: role-play behavior ===============================
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(14.5, 6.4), gridspec_kw={"width_ratios": [1.15, 1]})
    fig.patch.set_facecolor("#fcfcfb")
    for a in (ax, ax2):
        a.set_facecolor("#fcfcfb")
    xs = np.array([r3v[r] for r in sroles]); ys_ = np.array([r3t[r] for r in sroles])
    scats = [ROLE2CAT.get(r, "human (other)") for r in sroles]
    lim = max(xs.max(), ys_.max()) * 1.05
    ax.plot([0, lim], [0, lim], color="#bbb", lw=1, ls="--")
    for cat, col in COLORS.items():
        idx = [i for i, c in enumerate(scats) if c == cat]
        ax.scatter(xs[idx], ys_[idx], s=22, c=col, alpha=0.8, edgecolors="#fcfcfb", linewidths=0.5, zorder=3)
    mal = [i for i, r in enumerate(sroles) if r in MALEVOLENT]
    ax.scatter(xs[mal], ys_[mal], s=64, facecolors="none", edgecolors="#c0392b", linewidths=1.2, zorder=4, label="malevolent subset")
    d = ys_ - xs
    for i in np.argsort(d)[:8]:
        ax.annotate(sroles[i], (xs[i], ys_[i]), xytext=(4, -8), textcoords="offset points", fontsize=6.5, color="#52514e")
    ax.set_xlabel("vanilla: fraction of responses fully role-playing (judge score 3)", fontsize=9.5)
    ax.set_ylabel("SPP: fraction fully role-playing", fontsize=9.5)
    ax.set_title("(a) Full role-play rate per persona — points below the diagonal =\nSPP more resistant to inhabiting that persona", fontsize=10, loc="left")
    handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markersize=8, label=cat) for cat, c in COLORS.items()]
    handles.append(Line2D([0], [0], marker="o", color="w", markerfacecolor="none", markeredgecolor="#c0392b", markersize=9, label="malevolent subset"))
    ax.legend(handles=handles, loc="upper left", frameon=False, fontsize=8)

    cat_order = ["assistant-like", "professional", "human (other)", "AI / abstract", "non-human", "malevolent (subset)"]
    rel = [(M["cat_score3"][c]["spp"] - M["cat_score3"][c]["vanilla"]) / M["cat_score3"][c]["vanilla"] for c in cat_order]
    cols = [COLORS.get(c, "#c0392b") for c in cat_order]
    ax2.barh(range(len(cat_order))[::-1], [r * 100 for r in rel], color=cols, height=0.65)
    ax2.set_yticks(range(len(cat_order))[::-1])
    ax2.set_yticklabels([f"{c}  (n={M['cat_score3'][c]['n']})" for c in cat_order], fontsize=9)
    ax2.axvline(0, color="#52514e", lw=0.8)
    ax2.set_xlim(min(r * 100 for r in rel) * 1.18, 0.5)
    for k, c in enumerate(cat_order):
        ax2.text(rel[k] * 100 - 0.8, len(cat_order) - 1 - k, f"{rel[k]:+.0%}", va="center", ha="right", fontsize=9)
    ax2.set_xlabel("relative change in full-role-play rate, SPP vs vanilla (%)", fontsize=9.5)
    ax2.set_title("(b) Suppression scales with distance from the Assistant\n(refusals shift to score 1: 'I can't be that, but I can help')", fontsize=10, loc="left")
    for a in (ax, ax2):
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
    fig.suptitle("SPP changes role-play *willingness*, graded by persona type — while the persona geometry stays put",
                 fontsize=12.5, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(figdir / "spp_roleplay_behavior.png", dpi=170, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

    Path(args.outdir, "spp_vs_vanilla_metrics.json").write_text(json.dumps(M, indent=1))
    print(f"RSA={M['rsa_mid']:.3f}  PC1-loading r={M['pc_loading_corr'][0]:.3f}  "
          f"ordering rho={M['axis_ordering_spearman']:.3f}  disparity={M['procrustes_disparity']:.3f}")
    print(f"wrote {figdir}/spp_persona_movement.png, {figdir}/spp_roleplay_behavior.png, "
          f"{args.outdir}/spp_vs_vanilla_metrics.json")


if __name__ == "__main__":
    main()
