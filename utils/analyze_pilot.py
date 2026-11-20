#!/usr/bin/env python
"""Analysis of the style-preference pilot (questions from
utils/build_pilot_questionnaire.py, answers from the pilot page).

Per participant, a linear Bradley-Terry theta over the normalized style
features (laplace_posterior, prior N(0, I)) is fit on the train questions
("fit" in v1/v2) and scored on the held-out test questions, which never
enter any fit:
  personal   -- the participant's own theta
  shared     -- one theta fit on every OTHER participant's train answers
  prior      -- pick the more human-like option (higher log-prior)
From v3 on, the page shows the whole train block before any test question
and records each answer's position (n); a participant who answered a test
question before finishing the train block is reported (order_ok = False).
(The leave-one-question-out score over train + test of earlier versions is
gone: it fit on test answers.)
Also: repeat consistency (same code picked when the question comes back with
sides swapped -- the ceiling any model can reach), screen-left bias, per-axis
choice rates, pairwise answer agreement between participants and the cosine
similarity of their thetas. Means come with a 95% bootstrap over participants;
personal - shared is paired per participant.

Answers, either
  JSON: the responses collection as exported (a list of docs, a list of
        {"id", "data"} rows, or {uid: doc}); a doc is the page's record
        {code, answers: {qid: {picked, ...}}}, picked = index into q["codes"]
  dir:  one JSON doc per participant (results sent from pilot_offline.html)
  CSV:  columns participant, qid, picked

Usage:
    PYTHONPATH=. .venv/bin/python utils/analyze_pilot.py \
        --answers results/codebook/pilot_v3/responses \
        --out results/codebook/pilot_v3/analysis.json
"""
import argparse
import csv
import json
import os

import numpy as np

from src.preference.bayesian_query import laplace_posterior


def load_docs(path):
    """Page records from a JSON export, a single doc or a directory of docs."""
    if os.path.isdir(path):
        return [d for f in sorted(os.listdir(path)) if f.endswith(".json")
                for d in load_docs(os.path.join(path, f))]
    raw = json.load(open(path))
    if isinstance(raw, dict):
        raw = [raw] if "answers" in raw else [dict(v, _id=k) for k, v in raw.items()]
    return [r.get("data", r) for r in raw]


def load_answers(path):
    """Returns {participant: {qid: picked}}."""
    if path.endswith(".csv"):
        out = {}
        for row in csv.DictReader(open(path)):
            out.setdefault(row["participant"], {})[row["qid"]] = int(row["picked"])
        return out
    out = {}
    for n, doc in enumerate(load_docs(path)):
        name = doc.get("code") or doc.get("_id") or f"p{n}"
        while name in out:  # same code typed from two accounts
            name += "+"
        out[name] = {qid: int(a["picked"]) for qid, a in (doc.get("answers") or {}).items()}
    return out


def diffs(qs, picks, x):
    """Rows winner - loser for the answered questions."""
    return np.array([x[q["id"]][picks[q["id"]]] - x[q["id"]][1 - picks[q["id"]]] for q in qs])


def accuracy(theta, qs, picks, x):
    if not qs:
        return np.nan
    return float(np.mean([(diffs([q], picks, x)[0] @ theta > 0) for q in qs]))


def boot(v, rng, n=2000):
    v = np.asarray(v, float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return [np.nan] * 3
    m = v[rng.integers(len(v), size=(n, len(v)))].mean(1)
    return [float(v.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--questions", default="results/codebook/pilot_v3/questions.json")
    p.add_argument("--answers", required=True)
    p.add_argument("--min-fit", type=int, default=8, help="drop participants with fewer train answers")
    p.add_argument("--prior-std", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="results/codebook/pilot_v3/analysis.json")
    args = p.parse_args()
    rng = np.random.default_rng(args.seed)

    Q = json.load(open(args.questions))
    feats = Q["features"]
    D = len(feats)
    x = {q["id"]: np.array(q["x"]) for q in Q["questions"]}
    answers = load_answers(args.answers)
    alias = {"train": "fit"}
    split = lambda s, a: [q for q in Q["questions"] if alias.get(q["split"], q["split"]) == s and q["id"] in a]
    people = {k: a for k, a in answers.items() if len(split("fit", a)) >= args.min_fit}
    print(f"{len(answers)} participants, {len(people)} with >= {args.min_fit} fit answers")
    for k in sorted(set(answers) - set(people)):
        print(f"  dropped {k}: {len(answers[k])} answers")

    thetas, rows = {}, {}
    for k, a in people.items():
        thetas[k], _ = laplace_posterior(diffs(split("fit", a), a, x), D, args.prior_std)
    for k, a in people.items():
        fit, test = split("fit", a), split("test", a)
        others = [diffs(split("fit", b), b, x) for j, b in people.items() if j != k]
        shared = laplace_posterior(np.concatenate(others), D, args.prior_std)[0] if others else np.zeros(D)
        human = np.eye(D)[feats.index("human")]
        rep = [a[q["id"]] == 1 - a[q["repeat_of"]]  # sides swapped: same code = other index
               for q in split("repeat", a) if q["repeat_of"] in a]
        rows[k] = {
            "n": len(a), "n_fit": len(fit), "n_test": len(test),
            "personal": accuracy(thetas[k], test, a, x),
            "shared": accuracy(shared, test, a, x),
            "prior": accuracy(human, test, a, x),
            "repeat": float(np.mean(rep)) if rep else np.nan,
            "theta": thetas[k].round(3).tolist(),
            # per axis: rate of picking the option that is higher on that axis
            "axis_rate": {f: float(np.mean([x[q["id"]][a[q["id"]], i] > x[q["id"]][1 - a[q["id"]], i]
                                            for q in fit + test if q["axis"] == f]))
                          for i, f in enumerate(feats)},
        }

    # screen side bias and answer order need the page records (CSV has neither)
    if not args.answers.endswith(".csv"):
        docs = load_docs(args.answers)
        for n_doc, doc in enumerate(docs):
            name = doc.get("code") or doc.get("_id") or f"p{n_doc}"
            pos = {qid: r["n"] for qid, r in (doc.get("answers") or {}).items() if "n" in r}
            if name in rows and pos:
                blk = {q["id"]: alias.get(q["split"], q["split"]) for q in Q["questions"]}
                tr = [v for k, v in pos.items() if blk.get(k) == "fit"]
                te = [v for k, v in pos.items() if blk.get(k) in ("test", "repeat")]
                rows[name]["order_ok"] = bool(not tr or not te or max(tr) < min(te))
        left = [a["picked"] == a["left"] for d in docs for a in (d.get("answers") or {}).values() if "left" in a]
        ms = [a["ms"] for d in docs for a in (d.get("answers") or {}).values() if "ms" in a]
    else:
        left, ms = [], []

    names = sorted(rows)
    qids = [q["id"] for q in Q["questions"] if q["split"] != "repeat"]
    agree = [np.mean([people[i][q] == people[j][q] for q in qids if q in people[i] and q in people[j]])
             for n, i in enumerate(names) for j in names[n + 1:]]
    unit = {k: t / (np.linalg.norm(t) + 1e-12) for k, t in thetas.items()}
    cos = [float(unit[i] @ unit[j]) for n, i in enumerate(names) for j in names[n + 1:]]

    summary = {m: boot([rows[k][m] for k in names], rng) for m in ("personal", "shared", "prior", "repeat")}
    summary["personal-shared"] = boot([rows[k]["personal"] - rows[k]["shared"] for k in names], rng)
    summary["pair_agreement"] = [float(np.mean(agree)), float(np.min(agree)), float(np.max(agree))] if agree else None
    summary["theta_cos"] = [float(np.mean(cos)), float(np.min(cos)), float(np.max(cos))] if cos else None
    summary["left_rate"] = float(np.mean(left)) if left else None
    summary["median_ms"] = float(np.median(ms)) if ms else None

    print(f"\n{'participant':12s} {'n':>3s} {'pers':>5s} {'shared':>6s} {'prior':>5s} {'rep':>4s} {'order':>5s}  theta " + " ".join(feats))
    for k in names:
        r = rows[k]
        print(f"{k:12s} {r['n']:3d} {r['personal']:5.2f} {r['shared']:6.2f} {r['prior']:5.2f} "
              f"{r['repeat']:4.2f} {str(r.get('order_ok', '-')):>5s}  " + " ".join(f"{t:+.2f}" for t in r["theta"]))
    print("\nmean [95% CI over participants]")
    for m in ("personal", "shared", "personal-shared", "prior", "repeat"):
        v = summary[m]
        print(f"  {m:16s} {v[0]:+.3f} [{v[1]:+.3f}, {v[2]:+.3f}]")
    if agree:
        print(f"  answer agreement between participants  mean {summary['pair_agreement'][0]:.3f} "
              f"(min {summary['pair_agreement'][1]:.3f}, max {summary['pair_agreement'][2]:.3f})")
        print(f"  theta cosine between participants      mean {summary['theta_cos'][0]:+.3f} "
              f"(min {summary['theta_cos'][1]:+.3f}, max {summary['theta_cos'][2]:+.3f})")
    print("  per-axis rate of picking the higher option: " + ", ".join(
        f"{f} {np.mean([rows[k]['axis_rate'][f] for k in names]):.2f}" for f in feats))
    if left:
        print(f"  screen-left picks {summary['left_rate']:.3f}, median answer time {summary['median_ms'] / 1000:.1f} s")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump({"args": vars(args), "features": feats, "summary": summary, "participants": rows},
              open(args.out, "w"), indent=1)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
