import json, sys, numpy as np
d = sys.argv[1]; rng = np.random.default_rng(0)
def boot(v):
    m = v[rng.integers(len(v), size=(2000, len(v)))].mean(1)
    return v.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5)
for u in ("gauss", "sparse"):
    R = {q: json.load(open(f"{d}/gc_{u}_{q}.json")) for q in ("inter", "nointer", "all")}
    print(f"== {u}: situations conflict {R['inter']['n_situations']} / all {R['inter']['n_situations_all']}")
    for m in ("EUBO", "ActiveQuery", "Random"):
        for c in ("5", "10", "20"):
            a = {q: np.array(R[q]["per_user"][m][c])[:, 0] for q in R}
            line = f"  {m:11s} {c:>2s}q  inter {a['inter'].mean():.3f}  nointer {a['nointer'].mean():.3f}  all {a['all'].mean():.3f}"
            for q in ("nointer", "all"):
                mu, lo, hi = boot(a[q] - a["inter"])
                line += f"  {q}-inter {mu:+.3f} [{lo:+.3f},{hi:+.3f}]{'*' if lo > 0 or hi < 0 else ' '}"
            print(line)
