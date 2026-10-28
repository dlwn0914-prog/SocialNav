#!/bin/bash
# Overnight queue 10/28: waits for the per-candidate-endpoint (v3) extraction,
# then benchmarks it; runs the estimator ablation on the real-pedestrian user.
cd /home/nuri2/SocialNav
O=results/overnight_1028
V3=/home/nuri2/crowd_fm_ws/scenarios_v3/gc_all
V2=/home/nuri2/crowd_fm_ws/scenarios_v2/gc_all
FUT=/home/nuri2/crowd_fm_ws/scenarios_gc/ego_future_all.json
log() { echo "[$(date +%H:%M)] $*" >> $O/queue.log; }
B="PYTHONPATH=. .venv/bin/python utils/unified_query_benchmark.py --track real --user gc-ego --ego-future $FUT --moving-people --repeats 30"

# estimator ablation on the main setting does not need the GPU -- start now
log "start estimator ablation (v2 raw, shape+social)"
for e in mle map; do
  eval "$B --scenario-dir $V2 --candidates raw_candidates --answer shape --beta 17 --social --estimator $e --out $O/est_$e.json > $O/est_$e.log 2>&1 &"
done

while [ $(( $(ls $V3/scenario_*.json 2>/dev/null | wc -l) + $(grep -c FAILED $V3/run.log 2>/dev/null || echo 0) )) -lt 201 ]; do sleep 60; done
sleep 60
log "v3 extraction done: $(ls $V3/scenario_*.json | wc -l) files, $(grep -c FAILED $V3/run.log) failed"

PYTHONPATH=. .venv/bin/python - > $O/diversity_v3.txt 2>&1 <<'PY'
import glob, numpy as np
from utils.candidate_diversity_report import scene_stats
rows = [("v2 raw (pre-PRIEST)", "/home/nuri2/crowd_fm_ws/scenarios_v2/gc_all", "raw_candidates"),
        ("v2 refined (shared endpoint)", "/home/nuri2/crowd_fm_ws/scenarios_v2/gc_all", "candidates"),
        ("v3 raw (pre-PRIEST)", "/home/nuri2/crowd_fm_ws/scenarios_v3/gc_all", "raw_candidates"),
        ("v3 refined (per-candidate endpoint)", "/home/nuri2/crowd_fm_ws/scenarios_v3/gc_all", "candidates")]
print(f"{'set':>36} {'usable':>9} {'safe':>6} {'gap[m]':>7} {'lateral[m]':>11}")
for label, d, key in rows:
    st = [scene_stats(p, key, True) for p in sorted(glob.glob(d + "/scenario_*.json"))]
    u = [s for s in st if s[1] is not None]
    print(f"{label:>36} {len(u):>4}/{len(st):<4} {np.mean([s[0] for s in st]):>6.1f} "
          f"{np.mean([s[1] for s in u]):>7.2f} {np.mean([s[2] for s in u]):>11.2f}")
PY
log "diversity done"

for k in candidates raw_candidates; do
  eval "$B --scenario-dir $V3 --candidates $k --answer shape --beta 17 --social --out $O/v3_${k}_shape_social.json > $O/v3_${k}_shape_social.log 2>&1 &"
done
eval "$B --scenario-dir $V3 --candidates candidates --out $O/v3_candidates_ade_base.json > $O/v3_candidates_ade_base.log 2>&1 &"
PYTHONPATH=. .venv/bin/python utils/feature_upper_bound.py --scene-dir $V3 --ego-future $FUT --candidates candidates --answer shape --out $O/upper_v3_candidates_shape.json > $O/upper_v3_candidates_shape.txt 2>&1
log "v3 upper bound done"
wait
for f in $O/*.log; do [ "$f" = "$O/queue.log" ] || sed -n '/===/,$p' $f > ${f%.log}_summary.txt; done
log "all done"
