# Manuscript (OpenReview, 11/25)

LaTeX skeleton. There is no LaTeX on the lab server, so upload the `paper/` folder to Overleaf (main file `main.tex`).

`\todo{...}` (red) marks text still to write or check. Bibliography entries for Crowd-FM, PRIEST, UserAlign (AQ) and the crowd-navigation policies are still missing.

| File | Content | Numbers from |
|---|---|---|
| sections/abstract.tex | Abstract (11/18 version + DPO / generator results) | as below |
| sections/method.tex, sections/regret.tex | Codebook, safety filter, features, BT + EUBO, simple-regret definition | — |
| tables/regret.tex, tables/agreement.tex | Q1/Q2: Random / AQ / EUBO / DPO / DPO+EUBO questions | results/codebook/dpo/ |
| tables/generator.tex | Q3: Crowd-FM vs VQ-VAE × Random / AQ / EUBO | results/codebook/generator/ |
| tables/closed_loop.tex | Q4: closed-loop replay | results/codebook/closed_loop/*_safe_c0.5.json |
| figures/fig_regret.pdf | SR_n vs number of queries | utils/plot_regret_curves.py |
| sections/appendix.tex | Agreement table, ETH destination horizon | results/codebook/{dpo,style,dest8}/ |

Regenerate the tables and figure after a rerun:

    PYTHONPATH=. .venv/bin/python utils/make_paper_tables.py --out-dir paper/tables
    PYTHONPATH=. .venv/bin/python utils/plot_regret_curves.py --out paper/figures/fig_regret
