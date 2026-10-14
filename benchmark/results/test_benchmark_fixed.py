"""Offline regression tests against the installed official vLLM request code."""

import asyncio
import json
import os
import random
import unittest
from unittest.mock import patch

from aiohttp.helpers import get_env_proxy_for_url
from yarl import URL
from vllm.benchmarks.lib import endpoint_request_func as endpoint

from benchmark_fixed import configure_local_proxy_bypass, install_stream_fix


ORIGINAL_CHAT = endpoint.async_request_openai_chat_completions
ORIGINAL_COMPLETIONS = endpoint.async_request_openai_completions
PATCHED_COUNT = install_stream_fix(endpoint)
FIXED_CHAT = endpoint.async_request_openai_chat_completions
FIXED_COMPLETIONS = endpoint.async_request_openai_completions
TEXT = "hello  world 中文 尾部空格 "


def wire_data(chat=True):
    def choice(text):
        return {"delta": {"content": text}} if chat else {"text": text}

    events = [{"choices": [choice(text)]} for text in ["hello ", " world ", "中文 ", "尾部空格 "]]
    events.append({"choices": [], "usage": {"prompt_tokens": 264, "completion_tokens": 128}})
    return (
        "".join("data: " + json.dumps(event, ensure_ascii=False) + "\n\n" for event in events)
        + "data: [DONE]\n\n"
    ).encode()


class FakeResponse:
    def __init__(self, chunks, status=200):
        self.chunks = chunks
        self.content = self
        self.status = status
        self.reason = "Bad Request" if status != 200 else "OK"

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    async def iter_any(self):
        for chunk in self.chunks:
            yield chunk


class FakeSession:
    def __init__(self, chunks, status=200):
        self.chunks = chunks
        self.status = status
        self.payload = None

    def post(self, **kwargs):
        self.payload = kwargs["json"]
        return FakeResponse(self.chunks, self.status)


async def request(function, chunks, chat=True, status=200):
    suffix = "chat/completions" if chat else "completions"
    inputs = endpoint.RequestFuncInput(
        prompt="test prompt",
        api_url=f"http://127.0.0.1:8000/v1/{suffix}",
        prompt_len=256,
        output_len=128,
        model="Qwen/Qwen3-0.6B",
        ignore_eos=True,
    )
    session = FakeSession(chunks, status)
    result = await function(inputs, session)
    return result, session.payload


class ProxyTests(unittest.TestCase):
    def test_wildcard_misroutes_and_literal_bypass_fixes_aiohttp(self):
        env = {"http_proxy": "http://127.0.0.1:7897", "no_proxy": "127.*,localhost"}
        with patch.dict(os.environ, env, clear=True):
            url = URL("http://127.0.0.1:8000/v1/chat/completions")
            proxy, _ = get_env_proxy_for_url(url)
            self.assertEqual(str(proxy), "http://127.0.0.1:7897")
            configure_local_proxy_bypass()
            with self.assertRaises(LookupError):
                get_env_proxy_for_url(url)
            # External HTTP requests still use the existing proxy.
            proxy, _ = get_env_proxy_for_url(URL("http://example.com"))
            self.assertEqual(str(proxy), "http://127.0.0.1:7897")

    def test_existing_entries_and_idempotency(self):
        env = {"NO_PROXY": "internal.example", "no_proxy": "other.example"}
        once = configure_local_proxy_bypass(env)
        self.assertEqual(once, configure_local_proxy_bypass(env))
        self.assertEqual(env["no_proxy"], env["NO_PROXY"])
        for host in ["internal.example", "other.example", "127.0.0.1", "localhost", "::1"]:
            self.assertIn(host, env["NO_PROXY"].split(","))


class StreamTests(unittest.IsolatedAsyncioTestCase):
    async def assert_stream(self, chunks, chat=True):
        function = FIXED_CHAT if chat else FIXED_COMPLETIONS
        result, _ = await request(function, chunks, chat)
        self.assertTrue(result.success, result.error)
        self.assertEqual(result.generated_text, TEXT)
        self.assertEqual(result.output_tokens, 128)
        self.assertEqual(result.prompt_len, 264)

    async def test_original_fails_on_valid_split_prefix(self):
        wire = wire_data()
        chunks = [wire[:6], wire[6:]]
        original, _ = await request(ORIGINAL_CHAT, chunks)
        self.assertFalse(original.success)
        self.assertIn("JSONDecodeError", original.error)
        await self.assert_stream(chunks)

    async def test_all_single_split_positions(self):
        for chat in [True, False]:
            wire = wire_data(chat)
            for position in range(1, len(wire)):
                with self.subTest(chat=chat, split=position):
                    await self.assert_stream([wire[:position], wire[position:]], chat)

    async def test_one_byte_chunks_preserve_utf8_and_spaces(self):
        for chat in [True, False]:
            wire = wire_data(chat)
            await self.assert_stream([wire[i:i+1] for i in range(len(wire))], chat)

    async def test_complete_and_randomly_grouped_chunks(self):
        rng = random.Random(42)
        for chat in [True, False]:
            wire = wire_data(chat)
            await self.assert_stream([wire], chat)
            for _ in range(200):
                chunks = []
                position = 0
                while position < len(wire):
                    size = rng.randint(1, 150)
                    chunks.append(wire[position:position+size])
                    position += size
                await self.assert_stream(chunks, chat)

    async def test_request_payload_unchanged(self):
        for chat, original, fixed in [
            (True, ORIGINAL_CHAT, FIXED_CHAT),
            (False, ORIGINAL_COMPLETIONS, FIXED_COMPLETIONS),
        ]:
            wire = wire_data(chat)
            old_result, old_payload = await request(original, [wire], chat)
            new_result, new_payload = await request(fixed, [wire], chat)
            self.assertTrue(old_result.success)
            self.assertTrue(new_result.success)
            self.assertEqual(old_payload, new_payload)

    async def test_server_error_is_not_treated_as_success(self):
        result, _ = await request(FIXED_CHAT, [], status=400)
        self.assertFalse(result.success)
        self.assertEqual(result.error, "Bad Request")

    async def test_thirty_responses_at_each_concurrency(self):
        wire = wire_data()
        for concurrency in [1, 5, 10, 20, 30]:
            semaphore = asyncio.Semaphore(concurrency)

            async def one(index):
                async with semaphore:
                    split = 1 + index * 13 % (len(wire) - 1)
                    await self.assert_stream([wire[:split], wire[split:]])

            await asyncio.gather(*(one(index) for index in range(30)))

    async def test_registry_aliases_and_idempotency(self):
        self.assertEqual(PATCHED_COUNT, 2)
        self.assertIs(endpoint.ASYNC_REQUEST_FUNCS["openai-chat"], FIXED_CHAT)
        self.assertIs(endpoint.ASYNC_REQUEST_FUNCS["openai"], FIXED_COMPLETIONS)
        self.assertIs(endpoint.ASYNC_REQUEST_FUNCS["vllm"], FIXED_COMPLETIONS)
        self.assertEqual(install_stream_fix(endpoint), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
