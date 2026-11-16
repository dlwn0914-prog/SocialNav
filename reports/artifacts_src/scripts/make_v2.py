import json, re, sys
src, qpath, out = sys.argv[1:4]
h = open(src, encoding="utf-8").read()
h = h[h.index("<title>"):h.rindex("</body>")].rstrip() + "\n"
Q = json.load(open(qpath))
keep = ("id", "axis", "split", "people", "dest", "paths", "codes", "caption", "groups")
data = {"version": "pilot-v2", "dt": Q["dt"], "questions": [{k: q[k] for k in keep} for q in Q["questions"]]}
h = re.sub(r"const DATA = .*?;\n", lambda m: "const DATA = " + json.dumps(data, ensure_ascii=False, separators=(",", ":")) + ";\n", h, count=1)
n = len(data["questions"])
def rep(a, b):
    global h
    assert h.count(a) == 1, a
    h = h.replace(a, b)
rep("· 44문항 · 약 10분 ·", f"· {n}문항 · 약 13분 ·")
rep('<b id="qnum">1 / 44</b>', f'<b id="qnum">1 / {n}</b>')
rep("<li>그림은 위에서 내려다본 모습이고",
    "<li>사람들 사이를 지나가는 장면, 줄 서 있는 사람들, 둘러서서 대화하는 사람들, 나란히 걸어오는 일행, 앞에서 천천히 걷는 사람 등 여러 장면이 나옵니다. 장면 설명이 있으면 먼저 읽어 주세요.</li>\n<li>그림은 위에서 내려다본 모습이고")
rep("회색 점은 사람이고, 짧은 선은 걷는 방향입니다.</li>",
    "회색 점은 사람이고, 짧은 선은 걷는 방향입니다. 사람들을 잇는 선은 같은 줄이나 일행이라는 표시입니다.</li>")
rep('<span style="--c:var(--accent)">목적지 방향</span></div>',
    '<span style="--c:var(--accent)">목적지 방향</span></div>\n<p class="caption" id="caption" hidden></p>')
rep(".clock {", ".caption { margin:0; padding:8px 12px; border-left:3px solid var(--accent); background:var(--bg); font-weight:600; }\n.clock {")
# body-sized discs: person 0.2 m, robot 0.22 m radius
rep("g.arc(a, b, 0.25 * S, 0, 7)", "g.arc(a, b, 0.2 * S, 0, 7)")
rep("g.arc(ra, rb, 0.3 * S, 0, 7)", "g.arc(ra, rb, 0.22 * S, 0, 7)")
rep("  // people\n", """  // group links (same queue / party)
  const at = p => { const tt = Math.min(t, T_END); return X(p[0] + p[2] * tt, p[1] + p[3] * tt); };
  g.strokeStyle = css("--person"); g.globalAlpha = 0.6; g.lineWidth = 2 * dpr; g.setLineDash([]);
  for (const grp of q.groups || []) { g.beginPath(); grp.forEach((m, i) => (i ? g.lineTo : g.moveTo).call(g, ...at(q.people[m]))); g.stroke(); }
  g.globalAlpha = 1;
  // people
""")
rep("  $(\"bar\").style.width = `${100 * idx / state.order.length}%`;\n",
    "  $(\"bar\").style.width = `${100 * idx / state.order.length}%`;\n  $(\"caption\").textContent = q.caption || \"\"; $(\"caption\").hidden = !q.caption;\n")
open(out, "w", encoding="utf-8").write(h)
print(n, "questions", len(h) // 1024, "KB")
