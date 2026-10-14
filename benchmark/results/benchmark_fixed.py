#!/usr/bin/env python3
"""Run the official vLLM serving benchmark with two local client fixes.

This wrapper does not modify site-packages or server settings. It preserves
the official benchmark arguments, workload, timing, and metrics calculations.
"""

import inspect
import os
import sys


def configure_local_proxy_bypass(environ=None):
    """Use literal loopback hosts: urllib does not interpret '127.*' as a glob."""
    if environ is None:
        environ = os.environ
    entries = []
    for key in ("no_proxy", "NO_PROXY"):
        for entry in environ.get(key, "").split(","):
            entry = entry.strip()
            if entry and entry not in entries:
                entries.append(entry)
    for entry in ("127.0.0.1", "localhost", "::1"):
        if entry not in entries:
            entries.append(entry)
    value = ",".join(entries)
    environ["NO_PROXY"] = value
    environ["no_proxy"] = value
    return value


def install_stream_fix(endpoint):
    """Preserve bytes before official SSE buffering in text/chat backends.

    Check the exact operation before changing it, and update all aliases of
    each function in ASYNC_REQUEST_FUNCS. No HTTP status or error is suppressed.
    """
    replacements = {}
    for name in (
        "async_request_openai_completions",
        "async_request_openai_chat_completions",
    ):
        original = getattr(endpoint, name)
        if getattr(original, "_benchmark_sse_fixed", False):
            continue
        source = inspect.getsource(original)
        operation = "chunk_bytes = chunk_bytes.strip()"
        count = source.count(operation)
        if count == 0:
            continue
        if count != 1:
            raise RuntimeError(f"Unexpected implementation of {name}; refusing to patch")
        fixed_source = source.replace(
            operation,
            "# Preserve SSE delimiters and whitespace across network chunks.",
            1,
        )
        namespace = dict(endpoint.__dict__)
        exec(compile(fixed_source, f"<benchmark-stream-fix:{name}>", "exec"), namespace)
        patched = namespace[name]
        patched._benchmark_sse_fixed = True
        replacements[original] = patched
    for original, patched in replacements.items():
        setattr(endpoint, original.__name__, patched)
    for backend, function in list(endpoint.ASYNC_REQUEST_FUNCS.items()):
        if function in replacements:
            endpoint.ASYNC_REQUEST_FUNCS[backend] = replacements[function]
    return len(replacements)


def main():
    configure_local_proxy_bypass()
    from vllm.benchmarks.lib import endpoint_request_func as endpoint

    patched_count = install_stream_fix(endpoint)
    from vllm.benchmarks import serve
    from vllm.utils.argparse_utils import FlexibleArgumentParser

    parser = FlexibleArgumentParser(
        description="Official vLLM serving benchmark with local proxy/SSE fixes"
    )
    serve.add_cli_args(parser)
    args = parser.parse_args()
    if not args.save_result or not args.save_detailed:
        parser.error("Use --save-result --save-detailed to retain per-request errors")
    print(
        "[client fixes] Literal loopback hosts added to NO_PROXY/no_proxy; "
        f"SSE byte stripping removed from {patched_count} endpoint functions.",
        flush=True,
    )
    result = serve.main(args)
    failed = result.get("failed")
    if failed is None or failed != 0 or result.get("completed") != args.num_prompts:
        print(
            "[verification failed] Benchmark has failed or missing requests. "
            "Inspect the saved errors and console log.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
