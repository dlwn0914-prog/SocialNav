import sys
src, out = sys.argv[1], sys.argv[2]
L = open(src, encoding="utf-8").read().split("\n")
# L[0] = artifact skeleton head + <body>; page starts at line 2 (<title>)
head = L[1:60]            # title .. </header> (lines 2-60)
data = L[101]             # const DATA = ...
draw = L[102:171]         # const Q .. drawing helpers (lines 103-171)

intro = '''<section class="card" id="intro">
<h2>안내</h2>
<p>로봇이 사람들 사이를 지나 목적지 방향으로 가는 장면을 두 가지 방식으로 보여 드립니다. 두 경로 모두 안전합니다(사람과 0.5 m 이상 떨어져 지나감). 이 로봇을 쓰는 사람으로서 <b>어느 쪽 움직임이 더 마음에 드는지</b> 골라 주세요. 정답은 없습니다.</p>
<ul class="rules">
<li>그림은 위에서 내려다본 모습이고, 로봇은 아래에서 위쪽으로 출발합니다. 4.8초 동안의 움직임이 반복 재생됩니다.</li>
<li>점선은 로봇이 갈 경로, 초록 화살표는 목적지 방향입니다. 회색 점은 사람이고, 짧은 선은 걷는 방향입니다.</li>
<li>같은 장면이 두 번 나올 수 있습니다. 그때도 그 순간 끌리는 쪽을 고르면 됩니다.</li>
<li>참가자 코드(예: P03)만 기록합니다. 이름은 적지 마세요. 응답은 인터넷으로 전송되지 않고, 마지막 화면의 결과를 담당자에게 직접 보내 주시면 됩니다.</li>
<li>PC 브라우저(크롬, 엣지 등)에서 여는 것을 권장합니다. 중간에 창을 닫아도 같은 브라우저로 다시 열면 이어서 할 수 있습니다.</li>
</ul>
<div class="row">
<label for="pcode">참가자 코드</label>
<input type="text" id="pcode" maxlength="12" autocomplete="off" placeholder="P03" pattern="[A-Za-z0-9-]{1,12}">
<button class="primary" id="start">시작</button>
</div>
<p class="note" id="status">참가자 코드를 적고 시작을 누르세요.</p>
</section>'''

task = "\n".join(L[77:89]).replace('<button id="replay">처음부터 다시 재생</button><span class="note" id="saveState"></span>',
                                    '<button id="replay">처음부터 다시 재생</button>')

done = '''<section class="card" id="done" hidden>
<h2>끝났습니다. 감사합니다.</h2>
<p>응답 <b id="nDone">0</b>개를 마쳤습니다. 아래 결과를 <b>복사</b>하거나 <b>파일로 저장</b>해서 담당자에게 카카오톡이나 메일로 보내 주세요. 보내기 전에는 이 창을 닫지 마세요.</p>
<div class="row"><button class="primary" id="copy">결과 복사</button><button id="save">파일로 저장</button><span class="note" id="copyState"></span></div>
<textarea id="result" readonly rows="6" style="width:100%;font-family:var(--mono);font-size:.75rem;background:var(--bg);color:var(--fg);border:1px solid var(--line);border-radius:6px;padding:8px;box-sizing:border-box"></textarea>
</section>'''

flow = '''// ---- flow ----
const KEY = "nurilab-pilot-" + DATA.version;
const store = { get() { try { return JSON.parse(localStorage.getItem(KEY)); } catch (e) { return null; } },
                set(v) { try { localStorage.setItem(KEY, JSON.stringify(v)); } catch (e) {} } };
let code = "";
function showQuestion() {
  const q = current();
  if (!q) { finish(); return; }
  $("qnum").textContent = `${idx + 1} / ${state.order.length}`;
  $("bar").style.width = `${100 * idx / state.order.length}%`;
  t0 = performance.now(); shownAt = Date.now();
  cancelAnimationFrame(raf); raf = requestAnimationFrame(frame);
}
function answer(side) {
  const q = current(); if (!q) return;
  const [l, r] = sides(q), picked = side === 0 ? l : r;
  state.answers[q.id] = { picked, code: q.codes[picked], left: l, ms: Date.now() - shownAt };
  store.set({ code, ...state });
  idx++; showQuestion();
}
function finish() {
  cancelAnimationFrame(raf);
  $("task").hidden = true; $("done").hidden = false;
  $("nDone").textContent = Object.keys(state.answers).length;
  $("result").value = JSON.stringify({ code, version: DATA.version, flip: state.flip, answers: state.answers });
}
function start() {
  code = $("pcode").value.trim();
  if (!/^[A-Za-z0-9-]{1,12}$/.test(code)) { $("status").innerHTML = '<span class="warn">참가자 코드는 영문·숫자 12자 이내로 적어 주세요.</span>'; return; }
  const saved = store.get();
  state = saved && saved.code === code && saved.order ? { order: saved.order, flip: saved.flip, answers: { ...(saved.answers || {}) } }
                                                      : { ...plan("offline" + code), answers: {} };
  store.set({ code, ...state });
  idx = state.order.findIndex(id => !state.answers[id]); if (idx < 0) idx = state.order.length;
  $("intro").hidden = true; $("task").hidden = false; showQuestion();
}
$("start").onclick = start;
$("pcode").addEventListener("keydown", e => { if (e.key === "Enter") start(); });
$("pA").onclick = () => answer(0); $("pB").onclick = () => answer(1);
$("replay").onclick = () => { t0 = performance.now(); };
document.addEventListener("keydown", e => {
  if ($("task").hidden || e.target.tagName === "INPUT") return;
  if (e.key === "ArrowLeft") answer(0); else if (e.key === "ArrowRight") answer(1);
});
$("copy").onclick = () => {
  const ta = $("result");
  navigator.clipboard.writeText(ta.value).then(() => { $("copyState").textContent = "복사되었습니다."; },
    () => { ta.focus(); ta.select(); $("copyState").textContent = "선택되었습니다. Ctrl+C로 복사해 주세요."; });
};
$("save").onclick = () => {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([$("result").value], { type: "application/json" }));
  a.download = `pilot_${code}.json`; a.click();
};
{ const s = store.get(); if (s && s.code) { $("pcode").value = s.code; $("status").textContent = `이전 응답 ${Object.keys(s.answers || {}).length}개가 있습니다. 시작을 누르면 이어서 진행합니다.`; } }
'''

page = ["<!doctype html>", '<html lang="ko"><head><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width,initial-scale=1">'] + head[:1] + [
        "<style>body{margin:0}[hidden]{display:none!important}</style>"] + head[1:] + [
        intro, "", task, "", done, "</main>", "<script>", data] + draw + [flow, "</script>", "</body></html>"]
html = "\n".join(page).replace("NuRI Lab 내부 파일럿 · 44문항", "NuRI Lab 내부 파일럿 (오프라인판) · 44문항")
html = html.replace('<meta name="claude:viewport-fit-sides" content="cover">\n', "")
html = html.replace("let db = null, uid = null, docRef = null, state = null, idx = 0, shownAt = 0, busy = false;",
                    "let state = null, idx = 0, shownAt = 0;")
html = html.replace("<header>", "</head><body>\n<main>\n<header>", 1) if "<main>" not in "\n".join(head) else html.replace("<main>", "</head><body>\n<main>", 1)
open(out, "w", encoding="utf-8").write(html)
