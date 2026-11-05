#!/usr/bin/env bash
# Write the full_eval report for one experiment into results/full_eval_<exp>/.
# Usage: scripts/report_full_eval.sh <experiment id> [database]   (OUT_DIR=somewhere overrides the folder)
#   uv run a2e results list      # to find the experiment id
# The markdown report has every table. The CSV files are the same numbers cut by difficulty, for plotting.
set -euo pipefail
cd "$(dirname "$0")/.."
exp="${1:?usage: scripts/report_full_eval.sh <experiment id> [database]}"
db=()
[ -n "${2:-}" ] && db=(--db "$2")
out="${OUT_DIR:-results/full_eval_${exp}}"
mkdir -p "$out"
a2e() { uv run --quiet a2e "$@"; }

a2e results report "$exp" --out "$out/report.md" ${db[@]+"${db[@]}"}
a2e results jobs "$exp" --status failed ${db[@]+"${db[@]}"} > "$out/failed_jobs.txt" || true
# accuracy, speed and energy cut by what the task is
a2e results summary --exp "$exp" --by device,model,quant,grammar ${db[@]+"${db[@]}"} --csv "$out/by_model.csv" > /dev/null
a2e results summary --exp "$exp" --by device,model,quant,grammar,task_kind,category ${db[@]+"${db[@]}"} --csv "$out/by_category.csv" > /dev/null
a2e results summary --exp "$exp" --by device,model,quant,grammar,family,depth ${db[@]+"${db[@]}"} --csv "$out/by_family_depth.csv" > /dev/null
a2e results failures --exp "$exp" ${db[@]+"${db[@]}"} --csv "$out/failure_reasons.csv" > /dev/null
a2e results stages --exp "$exp" ${db[@]+"${db[@]}"} > "$out/stages.txt"
a2e results resources --exp "$exp" ${db[@]+"${db[@]}"} --csv "$out/resources.csv" > /dev/null
a2e results task-resources --exp "$exp" ${db[@]+"${db[@]}"} --csv "$out/per_task.csv" > /dev/null
# success per joule
a2e results efficiency summary --exp "$exp" ${db[@]+"${db[@]}"} --csv "$out/success_per_joule.csv" > /dev/null
a2e results efficiency frontier --exp "$exp" ${db[@]+"${db[@]}"} --csv "$out/pareto.csv" > /dev/null
a2e results efficiency grammar --exp "$exp" ${db[@]+"${db[@]}"} --csv "$out/grammar_effect.csv" > /dev/null
a2e results efficiency stages --exp "$exp" ${db[@]+"${db[@]}"} --csv "$out/energy_by_stage.csv" > /dev/null
a2e results efficiency budget --exp "$exp" ${db[@]+"${db[@]}"} --csv "$out/energy_budget.csv" > /dev/null
a2e results stability --exp "$exp" ${db[@]+"${db[@]}"} > "$out/stability.txt"

cat > "$out/README.txt" <<TXT
full_eval, experiment $exp
This was one measured run per job with no warm-up, so there is no repeat-to-repeat spread. Compare
categories and families by their totals, not by single cells (2 to 4 tasks each). Energy is a software
estimate with the idle baseline removed; check power_scope in report.md for what it includes. A configuration
with no energy readings has empty energy columns.
report.md          every table in one place
by_model.csv       one row per device, model, quant, grammar
by_category.csv    ... plus BFCL category
by_family_depth.csv ... plus custom family and chain depth
success_per_joule.csv, pareto.csv, grammar_effect.csv, energy_by_stage.csv, energy_budget.csv
resources.csv      power, memory, CPU and thermal envelope per configuration
per_task.csv       energy and time of every task
failure_reasons.csv  why tasks failed, by model, category and reason
failed_jobs.txt, stability.txt, stages.txt
TXT
echo "wrote $out/"
ls "$out"
