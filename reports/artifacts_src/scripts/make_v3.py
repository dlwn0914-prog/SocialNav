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
# destination: far outside the view (15 m+), the arrow only gives its direction (11/20)
rep("초록 화살표는 목적지 방향입니다.",
    "초록 화살표는 목적지 방향입니다. 목적지는 화면 밖 멀리(15 m 이상 앞)에 있고, 화살표는 방향만 나타냅니다. 화살표 끝이 도착 지점이 아닙니다.")
rep('g.strokeStyle = css("--accent"); g.globalAlpha = 0.55; g.lineWidth = 2 * dpr; g.setLineDash([2 * dpr, 4 * dpr]);',
    'g.strokeStyle = css("--accent"); g.globalAlpha = 0.85; g.lineWidth = 2.5 * dpr; g.setLineDash([10 * dpr, 7 * dpr]);')
rep("""  g.lineTo(d1x - 12 * dpr * Math.cos(ang - 0.4), d1y - 12 * dpr * Math.sin(ang - 0.4));
  g.lineTo(d1x - 12 * dpr * Math.cos(ang + 0.4), d1y - 12 * dpr * Math.sin(ang + 0.4)); g.fill(); g.globalAlpha = 1;""",
    """  g.lineTo(d1x - 18 * dpr * Math.cos(ang - 0.4), d1y - 18 * dpr * Math.sin(ang - 0.4));
  g.lineTo(d1x - 18 * dpr * Math.cos(ang + 0.4), d1y - 18 * dpr * Math.sin(ang + 0.4)); g.fill(); g.globalAlpha = 1;
  g.font = `600 ${12 * dpr}px system-ui, sans-serif`; g.textAlign = d1x < cv.width / 2 ? "left" : "right"; g.textBaseline = "top";
  g.fillText("목적지 방향 (15 m+ 앞)", d1x + (d1x < cv.width / 2 ? 10 : -10) * dpr, Math.max(4 * dpr, d1y - 6 * dpr));""")
# robot path: solid and stronger than the dashed destination arrow, so the two are not confused
rep('g.strokeStyle = css("--robot"); g.lineWidth = 1.5 * dpr; g.setLineDash([5 * dpr, 4 * dpr]); g.globalAlpha = 0.6;',
    'g.strokeStyle = css("--robot"); g.lineWidth = 2.5 * dpr; g.setLineDash([]); g.globalAlpha = 0.85;')
rep("점선은 로봇이 갈 경로", "파란 선은 로봇이 갈 경로")
open(out, "w", encoding="utf-8").write(h)
print(n, "questions", len(h) // 1024, "KB")
