import re, sys
h = open("body.html").read()
body = re.search(r"<body>(.*)</body>", h, re.S).group(1)
# keep pandoc title block (title, author, abstract)
# paragraph headings: no x.0.0.n numbers
body = re.sub(r'<span class="header-section-number">\d+(?:\.\d+){3}</span>\s*', "", body)
# appendix = section 7 -> A
body = re.sub(r'(header-section-number">)7(\.\d+)?<', lambda m: m.group(1) + "A" + (m.group(2) or "") + "<", body)
body = re.sub(r'(data-reference="app:[^"]*">)7\.', r"\1A.", body)
# equation numbers (equation envs in order)
eqs = ["eq:safe", None, "eq:exec", "eq:regret", "eq:sr"]
n = iter(range(1, 99))
def num(m):
    k = next(n)
    lab = eqs[k - 1]
    return f'<div class="eq"{f" id={chr(34)}{lab}{chr(34)}" if lab else ""}>{m.group(0)}<span class="eqno">({k})</span></div>'
body = re.sub(r'<math display="block".*?</math>', num, body, flags=re.S)
for i, lab in enumerate(eqs, 1):
    if lab:
        body = re.sub(rf'(data-reference-type="eqref" data-reference="{lab}">)\[{lab}\]', rf"\1({i})", body)
        body = re.sub(rf'(data-reference-type="ref" data-reference="{lab}">)\[{lab}\]', rf"\g<1>{i}", body)
# captions
k = iter(range(1, 99)); body = re.sub(r"<figcaption>", lambda m: f"<figcaption><b>Figure {next(k)}.</b> ", body)

body = body.replace("<strong>[TODO:", '<strong class="todo">[TODO:')
# run-in paragraph headings (\\paragraph)
body = re.sub(r'<h([45])[^>]*>(.*?)</h\1>\s*<p>', r'<p><strong class="runin">\2</strong> ', body, flags=re.S)
# table captions above the scroll box, not inside the table
body = re.sub(r'<table([^>]*)>\s*<caption>(.*?)</caption>', r'<p class="tcap">\2</p><table\1>', body, flags=re.S)
body = re.sub(r"<table", '<div class="tw"><table', body); body = body.replace("</table>", "</table></div>")
k = iter(range(1, 99)); body = re.sub(r'<p class="tcap">', lambda m: f'<p class="tcap"><b>Table {next(k)}.</b> ', body)
ntodo = body.count('class="todo"')
tpl = open("page.tpl").read()
open(sys.argv[1], "w").write(tpl.replace("{{BODY}}", body).replace("{{NTODO}}", str(ntodo)).replace("{{COMMIT}}", sys.argv[2]))
print("todos", ntodo, "size", len(tpl) + len(body))
