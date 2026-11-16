#!/bin/bash
# 11/3: feature-criteria rerun. Main setting (v3 refined candidates, shape answer, beta 17, moving people, 30 reps)
# with fixed6 (10/28 social set, shared start waypoint dropped) and draft5 (criteria-based set), + upper bound/shared CV.
cd /home/nuri2/SocialNav
O=results/features_1103
V3=/home/nuri2/crowd_fm_ws/scenarios_v3/gc_all
FUT=/home/nuri2/crowd_fm_ws/scenarios_gc/ego_future_all.json
B="PYTHONPATH=. .venv/bin/python utils/unified_query_benchmark.py --track real --user gc-ego --ego-future $FUT --moving-people --repeats 30 --scenario-dir $V3 --candidates candidates --answer shape --beta 17"
for fs in fixed6 draft5; do
  eval "$B --features $fs --out $O/v3_shape_$fs.json > $O/v3_shape_$fs.log 2>&1 &"
done
PYTHONPATH=. .venv/bin/python utils/feature_upper_bound.py --scene-dir $V3 --ego-future $FUT --candidates candidates --answer shape --out $O/upper_v3_shape.json > $O/upper_v3_shape.txt 2>&1
wait
for fs in fixed6 draft5; do sed -n '/===/,$p' $O/v3_shape_$fs.log > $O/v3_shape_${fs}_summary.txt; done
echo done
