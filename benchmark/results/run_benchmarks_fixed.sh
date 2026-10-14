#!/usr/bin/env bash
set -euo pipefail

report_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd -- "$report_dir/../.." && pwd)"
python_bin="$project_dir/.venv/bin/python"
concurrencies=("$@")
if [ "${#concurrencies[@]}" -eq 0 ]; then
    concurrencies=(5 10 20 30)
fi
for concurrency in "${concurrencies[@]}"; do
    if ! [[ "$concurrency" =~ ^[1-9][0-9]*$ ]]; then
        printf 'Invalid concurrency: %s\n' "$concurrency" >&2
        exit 2
    fi
done
run_dir="$(mktemp -d "$report_dir/fixed-$(date +%Y%m%d-%H%M%S)-XXXXXX")"

# Use the cached model/tokenizer; set HF_HUB_OFFLINE=0 to allow downloads.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export PYTHONDONTWRITEBYTECODE=1

printf 'Results and complete logs: %s\n' "$run_dir"

# A direct health check gives a clear error before loading the tokenizer.
"$python_bin" - <<'PY' 2>&1 | tee "$run_dir/health.log"
from urllib.error import URLError
from urllib.request import ProxyHandler, build_opener
try:
    with build_opener(ProxyHandler({})).open(
        "http://127.0.0.1:8000/health", timeout=10
    ) as response:
        if response.status != 200:
            raise SystemExit(f"Unexpected health status: {response.status}")
except URLError as exc:
    raise SystemExit(f"Cannot directly reach vLLM at 127.0.0.1:8000: {exc}")
print("Direct server health check: OK")
PY

any_failed=0
for concurrency in "${concurrencies[@]}"; do
    if "$python_bin" "$report_dir/benchmark_fixed.py" \
        --backend openai-chat \
        --model Qwen/Qwen3-0.6B \
        --host 127.0.0.1 \
        --port 8000 \
        --endpoint /v1/chat/completions \
        --dataset-name random \
        --random-input-len 256 \
        --random-output-len 128 \
        --num-prompts 30 \
        --max-concurrency "$concurrency" \
        --request-rate inf \
        --ignore-eos \
        --save-result \
        --save-detailed \
        --result-dir "$run_dir" \
        --result-filename "c${concurrency}.json" \
        2>&1 | tee "$run_dir/c${concurrency}.log"; then
        printf 'Concurrency %s: all requests succeeded.\n' "$concurrency"
    else
        any_failed=1
        printf 'Concurrency %s: run failed; see the saved log.\n' "$concurrency" >&2
    fi
done

for concurrency in "${concurrencies[@]}"; do
    if [ ! -s "$run_dir/c${concurrency}.json" ]; then
        printf 'Missing result for concurrency %s; no complete report generated.\n' "$concurrency" >&2
        exit 1
    fi
done
"$python_bin" "$report_dir/generate_report.py" --result-dir "$run_dir" \
    2>&1 | tee "$run_dir/report-generation.log"
printf 'Report: %s/benchmark_report.md\n' "$run_dir"
exit "$any_failed"
