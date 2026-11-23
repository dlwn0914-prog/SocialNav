#!/bin/sh
# Manuscript (paper/) -> one-page HTML preview for the artifact. Needs pandoc >= 3
# (no system pandoc on the lab server: pip install pypandoc_binary, then PANDOC=<its path>).
# Usage: PANDOC=/path/to/pandoc sh reports/artifacts_src/paper_html/build.sh
# Korean translation: PAPER=paper_ko OUT=index_ko.html TPL=page_ko.tpl sh ...
set -e
D=$(cd "$(dirname "$0")" && pwd); R=$D/../../..
cd "$R/${PAPER:-paper}"
python3 "$D/flatten.py" . > "$D/flat.tex"
"${PANDOC:-pandoc}" "$D/flat.tex" --resource-path=. --citeproc --bibliography=refs.bib \
  --mathml --number-sections -t html5 --embed-resources -s -o "$D/body.html"
cd "$D" && python3 post.py "${OUT:-index.html}" "$(git -C "$R" rev-parse --short HEAD)" "${TPL:-page.tpl}" && rm flat.tex body.html
