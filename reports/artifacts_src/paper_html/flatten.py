import re, sys, os
root = sys.argv[1]
def load(p):
    p = p if p.endswith(".tex") else p + ".tex"
    s = open(os.path.join(root, p)).read()
    s = re.sub(r"(?<!\\)%.*", "", s)
    return re.sub(r"\\input\{([^}]*)\}", lambda m: load(m.group(1)), s)
def unwrap(s, cmd, repl):
    # replace \cmd{arg} with repl(arg), brace-matched
    out, i = [], 0
    while True:
        j = s.find(cmd + "{", i)
        if j < 0: return "".join(out) + s[i:]
        k, depth = j + len(cmd), 0
        while True:
            depth += {"{": 1, "}": -1}.get(s[k], 0)
            if depth == 0: break
            k += 1
        out += [s[i:j], repl(s[j + len(cmd) + 1:k])]
        i = k + 1
s = load("main")
s = s.replace(r"\newcommand{\todo}[1]{\textcolor{red}{[TODO: #1]}}", "")
s = unwrap(s, r"\todo", lambda a: r"\textbf{[TODO: " + a + "]}")
s = re.sub(r"\\resizebox\{[^}]*\}\{!\}\{", r"{", s)
s = re.sub(r"\\cmidrule\([a-z]*\)\{[^}]*\}", "", s)
s = s.replace(r"\scriptsize", "").replace(r"\setlength{\tabcolsep}{3pt}", "")
def png(m):  # HTML cannot show PDF figures; fig_regret.png lives in reports/
    f = m.group(1) + ".png"
    return f if os.path.exists(os.path.join(root, f)) else "../reports/" + os.path.basename(f)
s = re.sub(r"(figures/fig_\w+)\.pdf", png, s)
sys.stdout.write(s)
