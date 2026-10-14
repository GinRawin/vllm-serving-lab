"""Check the local Qwen server, including top-k/top-p sampling."""

import json
from urllib.request import ProxyHandler, Request, build_opener


BASE_URL = "http://127.0.0.1:8000"
MODEL = "Qwen/Qwen3-0.6B"
# Local requests must not go through the shell's HTTP proxy.
client = build_opener(ProxyHandler({}))


def request(path, payload=None):
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = Request(
        BASE_URL + path,
        data=data,
        headers={"Content-Type": "application/json"},
    )
    with client.open(req, timeout=120) as response:
        body = response.read()
        return json.loads(body) if body else None


def main():
    request("/health")
    models = request("/v1/models")
    assert MODEL in {item["id"] for item in models["data"]}, models
    print("Health and model listing: OK")

    for label, prompt, params in [
        ("greedy", "请用一句话介绍你自己。", {"temperature": 0}),
        (
            "top-k/top-p",
            "用一句中文打招呼。",
            {"temperature": 0.7, "top_p": 0.9, "top_k": 20},
        ),
        ("English", "What is two plus three?", {"temperature": 0}),
    ]:
        result = request(
            "/v1/chat/completions",
            {
                "model": MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 128,
                "chat_template_kwargs": {"enable_thinking": False},
                **params,
            },
        )
        choice = result["choices"][0]
        answer = choice["message"]["content"]
        assert answer and answer.strip(), result
        assert choice["finish_reason"] == "stop", result
        print(f"{label}: {answer.strip()!r}; usage={result['usage']}")


if __name__ == "__main__":
    main()
