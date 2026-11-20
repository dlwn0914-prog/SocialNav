"""Pilot page v2 -> v3 (20 questions, train block first, answer positions recorded).
Usage: python make_v3.py <v2 page .html> <pilot_v3/questions.json> <out .html>  (online or offline page)"""
import json, re, sys
src, qpath, out = sys.argv[1:4]
h = open(src, encoding="utf-8").read()
Q = json.load(open(qpath))
keep = ("id", "axis", "split", "block", "people", "dest", "paths", "codes", "caption", "groups")
data = {"version": "pilot-v3", "dt": Q["dt"], "questions": [{k: q[k] for k in keep if k in q} for q in Q["questions"]]}
h = re.sub(r"const DATA = .*?;\n", lambda m: "const DATA = " + json.dumps(data, ensure_ascii=False, separators=(",", ":")) + ";\n", h, count=1)
n = len(data["questions"])
def rep(a, b):
    global h
    assert h.count(a) == 1, a
    h = h.replace(a, b)
rep("· 60문항 · 약 13분 ·", f"· {n}문항 · 약 5분 ·")
rep('<b id="qnum">1 / 60</b>', f'<b id="qnum">1 / {n}</b>')
# train block (block 0) always before the test block (block 1); shuffled within each block
rep("""  const r = rng(hash(seedStr)), order = Q.map((_, i) => i);
  for (let i = order.length - 1; i > 0; i--) { const j = Math.floor(r() * (i + 1)); [order[i], order[j]] = [order[j], order[i]]; }
""", """  const r = rng(hash(seedStr));
  const shuf = a => { for (let i = a.length - 1; i > 0; i--) { const j = Math.floor(r() * (i + 1)); [a[i], a[j]] = [a[j], a[i]]; } return a; };
  const order = [0, 1].flatMap(b => shuf(Q.map((_, i) => i).filter(i => Q[i].block === b)));
""")
# answer position, so the analysis can check that no test answer came before the end of the train block
h, k = re.subn(r"(\{ picked, code: q\.codes\[picked\], left: l, ms: Date\.now\(\) - shownAt)",
               r"\1, n: Object.keys(state.answers).length", h)
assert k == 1, k
open(out, "w", encoding="utf-8").write(h)
print(n, "questions", len(h) // 1024, "KB")
