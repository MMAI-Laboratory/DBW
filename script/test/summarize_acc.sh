#!/usr/bin/env bash
set -u
cd "$(dirname "$0")/../.."

RESULT=${RESULT:-output/acc_results.jsonl}
LOGDIR=${LOGDIR:-output/acc_logs}
PY=${PY:-python}

cat "$LOGDIR"/*.json > "$RESULT" 2>/dev/null

"$PY" - "$RESULT" <<'PY'
import json, sys

ORDER = [
    "dbw-Llama-7b-w1.61g128", "dbw-Llama-2-7b-w1.61g128",
    "dbw-Llama-3-8b-w1.61g128",
    "dbw-Llama-65b-w1.61g128", "dbw-Llama-2-70b-w1.61g128",
    "dbw-gemma-7b-w1.61g128", "dbw-mistral-7b-w1.61g128", "dbw-qwen2.5-7b-w1.61g128",
    "dbw-Llama-2-7b-chat-w1.61g128",
    "dbw-DeepSeek-R1-Distill-Llama-8B-w1.61g128",
]
SHORT = {
    "piqa": "piqa", "arc_easy": "arc-e", "hellaswag": "hella",
    "winogrande": "wino", "race": "race", "arc_challenge": "arc-c",
    "lambada_openai": "lmb-o", "lambada_standard": "lmb-s",
}

rows = {}
for line in open(sys.argv[1]):
    if line.strip():
        r = json.loads(line)
        rows[r["tag"]] = r
if not rows:
    print("no accuracy results yet"); raise SystemExit

seen = [t for t in ORDER if t in rows] + [t for t in sorted(rows) if t not in ORDER]
tasks = list(rows[seen[0]]["acc"])
w = max(len(t) for t in seen)

head = " | ".join(f"{SHORT.get(t, t):>5}" for t in tasks)
print(f"| {'model':<{w}} | {head} |   avg |")
print(f"|{'-'*(w+2)}|" + "|".join("------:" for _ in tasks) + "|------:|")
for t in seen:
    a = rows[t]["acc"]
    cells = " | ".join(f"{a[k]*100:5.2f}" for k in tasks)
    print(f"| {t:<{w}} | {cells} | {rows[t]['acc_avg']*100:5.2f} |")
print(f"\n{len(seen)} models — metric is `acc` (not acc_norm), 0-shot")
missing = [t for t in ORDER if t not in rows]
if missing:
    print("not yet evaluated: " + ", ".join(missing))
PY
echo "raw -> $RESULT ; logs -> $LOGDIR"
