#!/usr/bin/env python3
"""Role-play capability gate for the SPP models.

These are small, lightly post-trained models. Before spending a night on the full pipeline we
check they can actually leave the assistant persona — without enough "fully role-playing"
(score=3) responses, the axis computation is starved of data.

Reads judge scores for a set of probe roles and reports, per model, the fraction of roles that
would yield a usable role vector at the full run's scale. The cross-model comparison of role-play
rates is itself a result: does persona pretraining make a model less willing to role-play?

Exit code 0 = GO, 1 = PARTIAL (lower min_count), 2 = NO-GO.
"""
import argparse
import json
from pathlib import Path


def load_scores(scores_dir):
    """role -> list of int scores."""
    out = {}
    for f in sorted(Path(scores_dir).glob("*.json")):
        data = json.load(open(f))
        vals = data.values() if isinstance(data, dict) else data
        scores = []
        for v in vals:
            if isinstance(v, dict):
                v = v.get("score")
            if isinstance(v, (int, float)):
                scores.append(int(v))
        if scores:
            out[f.stem] = scores
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores_dir", required=True)
    ap.add_argument("--key", required=True)
    ap.add_argument("--out", required=True, help="markdown report path")
    ap.add_argument("--full_scale", type=int, default=600, help="responses/role in the full run")
    ap.add_argument("--min_count", type=int, default=25, help="min score-3 needed at full scale")
    args = ap.parse_args()

    scores = load_scores(args.scores_dir)
    scores.pop("default", None)
    if not scores:
        print("NO SCORES FOUND", flush=True)
        return 2

    lines = [f"# {args.key} — role-play gate", "",
             "| role | n | score 0 | 1 | 2 | 3 | %3 | viable |", "|---|---|---|---|---|---|---|---|"]
    viable_roles, rates = 0, []
    for role, s in sorted(scores.items()):
        n = len(s)
        c = {k: s.count(k) for k in (0, 1, 2, 3)}
        rate = c[3] / n if n else 0.0
        rates.append(rate)
        # viable if, projected to the full run's scale, it clears min_count
        viable = (rate * args.full_scale) >= args.min_count
        viable_roles += bool(viable)
        lines.append(f"| {role} | {n} | {c[0]} | {c[1]} | {c[2]} | {c[3]} | {rate:.1%} | {'yes' if viable else 'no'} |")

    frac = viable_roles / len(scores)
    mean_rate = sum(rates) / len(rates)
    any_roleplay = sum(1 for s in scores.values() if (s.count(2) + s.count(3)) / len(s) >= 0.05) / len(scores)

    if frac >= 0.70:
        verdict, code = "GO", 0
    elif frac >= 0.30:
        verdict, code = "PARTIAL", 1
    else:
        verdict, code = "NO-GO", 2

    summary = {"key": args.key, "verdict": verdict, "viable_fraction": frac,
               "mean_score3_rate": mean_rate, "any_roleplay_fraction": any_roleplay,
               "n_probe_roles": len(scores)}
    lines = [f"**Verdict: {verdict}** — {viable_roles}/{len(scores)} probe roles viable "
             f"({frac:.0%}); mean score-3 rate {mean_rate:.1%}; "
             f"{any_roleplay:.0%} of roles show any role-play (score≥2 on ≥5% of responses).", ""] + lines
    Path(args.out).write_text("\n".join(lines))
    Path(args.out).with_suffix(".json").write_text(json.dumps(summary, indent=2))
    print(f"[{args.key}] gate: {verdict} ({frac:.0%} viable, mean score-3 {mean_rate:.1%})")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
