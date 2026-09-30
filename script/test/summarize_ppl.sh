#!/usr/bin/env bash
set -u
cd "$(dirname "$0")/../.."

RESULT=${RESULT:-output/ppl_results.jsonl}
LOGDIR=${LOGDIR:-output/eval_logs}
PY=${PY:-python}

cat "$LOGDIR"/*.json > "$RESULT" 2>/dev/null

"$PY" - "$RESULT" <<'PY'
import json, sys, collections

ORDER = [
    "dbw-Llama-7b-w1.61g128", "dbw-Llama-13b-w1.61g128",
    "dbw-Llama-30b-w1.61g128", "dbw-Llama-65b-w1.61g128",
    "dbw-Llama-2-7b-w0.5g128", "dbw-Llama-2-7b-w1g128",
    "dbw-Llama-2-7b-w1.61g128", "dbw-Llama-2-13b-w1.61g128",
    "dbw-Llama-2-70b-w1.61g128",
    "dbw-Llama-3-8b-w1.61g128",
    "dbw-gemma-7b-w1.61g128", "dbw-mistral-7b-w1.61g128",
    "dbw-qwen2.5-7b-w1.61g128",
    "dbw-Llama-2-7b-chat-w1.61g128",
    "dbw-DeepSeek-R1-Distill-Llama-8B-w1.61g128",
]

rows = {}
for line in open(sys.argv[1]):
    if line.strip():
        r = json.loads(line)
        rows[r["tag"]] = r

seen = [t for t in ORDER if t in rows] + [t for t in sorted(rows) if t not in ORDER]
w = max((len(t) for t in seen), default=10)

print(f"| {'model':<{w}} | wikitext2 |      c4 |")
print(f"|{'-'*(w+2)}|----------:|--------:|")
for t in seen:
    p = rows[t]["ppl"]
    wt = f"{p['wikitext2']:.2f}" if "wikitext2" in p else "-"
    c4 = f"{p['c4']:.2f}" if "c4" in p else "-"
    print(f"| {t:<{w}} | {wt:>9} | {c4:>7} |")
print(f"\n{len(seen)} models")
missing = [t for t in ORDER if t not in rows]
if missing:
    print("not yet evaluated: " + ", ".join(missing))
PY
echo "raw -> $RESULT ; logs -> $LOGDIR"
