import asyncio
import json
import unittest
from unittest.mock import patch

import httpx
import litellm

from magic_hour_litellm import (
    MagicHourError, MagicHourImageProvider, MagicHourProxyHook,
    aimage_generation, image_generation, register,
)

SYNC_CLIENT = httpx.Client
ASYNC_CLIENT = httpx.AsyncClient
PROJECT_ID = "cmagicHourExistingProject"
SIGNED_URL = "https://media.magichour.ai/image.png?signature=keep%2Fexact&expires=123"


def complete(**updates):
    return {
        "id": PROJECT_ID, "status": "complete", "credits_charged": 5,
        "downloads": [{"url": SIGNED_URL, "expires_at": "2026-10-25T06:00:00Z"}],
        **updates,
    }


class ProviderTests(unittest.TestCase):
    def setUp(self):
        self.requests = []
        self.old_map = litellm.custom_provider_map[:]
        register()

    def tearDown(self):
        litellm.custom_provider_map = self.old_map

    def clients(self, handler):
        def capture(request):
            self.requests.append(request)
            return handler(request)
        transport = httpx.MockTransport(capture)
        return (
            patch("magic_hour_litellm.httpx.Client", side_effect=lambda **kwargs: SYNC_CLIENT(transport=transport, **kwargs)),
            patch("magic_hour_litellm.httpx.AsyncClient", side_effect=lambda **kwargs: ASYNC_CLIENT(transport=transport, **kwargs)),
        )

    def call(self, **kwargs):
        return image_generation(
            model="magichour/z-image-turbo", prompt="A ceramic mug", api_key="not-a-real-key",
            num_retries=0, timeout=2, **kwargs,
        )

    def test_real_litellm_dispatch_create_once_preserves_url_and_credits(self):
        polls = 0
        def handler(request):
            nonlocal polls
            if request.method == "POST":
                self.assertEqual(request.url.path, "/v1/ai-image-generator")
                self.assertEqual(json.loads(request.content), {
                    "image_count": 1, "model": "z-image-turbo", "aspect_ratio": "1:1",
                    "resolution": "640px", "style": {"prompt": "A ceramic mug", "tool": "general"},
                })
                return httpx.Response(200, json={"id": PROJECT_ID, "credits_charged": 5})
            self.assertEqual(request.url.path, f"/v1/image-projects/{PROJECT_ID}")
            polls += 1
            return httpx.Response(200, json=complete(status="rendering") if polls == 1 else complete())
        sync, _ = self.clients(handler)
        with sync, patch("magic_hour_litellm.time.sleep"):
            result = self.call(n=1, size="auto", response_format="url")
        self.assertEqual([r.method for r in self.requests], ["POST", "GET", "GET"])
        self.assertEqual(result.data[0].url, SIGNED_URL)
        self.assertEqual(result.data[0].provider_specific_fields["project_id"], PROJECT_ID)
        self.assertEqual(result.data[0].provider_specific_fields["credits_charged"], 5)
        # LiteLLM adds zero-token accounting for this image response. It is
        # separate from the five Magic Hour credits above, not a USD price.
        self.assertEqual(result.usage.total_tokens, 0)
        self.assertTrue(all(r.url.host == "api.magichour.ai" for r in self.requests))

    def test_real_async_dispatch_and_provider_resolution(self):
        def handler(request):
            if request.method == "POST":
                self.assertEqual(json.loads(request.content)["resolution"], "1k")
                return httpx.Response(200, json={"id": PROJECT_ID, "credits_charged": 10})
            return httpx.Response(200, json=complete(credits_charged=10))
        _, asynchronous = self.clients(handler)
        async def run():
            return await aimage_generation(
                model="magichour/z-image-turbo", prompt="A ceramic mug", api_key="not-a-real-key",
                n=1, size="auto", magic_hour_resolution="1k", num_retries=0, timeout=2,
            )
        with asynchronous:
            result = asyncio.run(run())
        self.assertEqual(result.data[0].url, SIGNED_URL)
        self.assertEqual([r.method for r in self.requests], ["POST", "GET"])

    def test_omitted_options_survive_real_sync_and_async_dispatch(self):
        def handler(request):
            if request.method == "POST":
                body = json.loads(request.content)
                self.assertEqual(body["image_count"], 1)
                self.assertEqual(body["resolution"], "640px")
                return httpx.Response(200, json={"id": PROJECT_ID, "credits_charged": 5})
            return httpx.Response(200, json=complete())
        sync, asynchronous = self.clients(handler)
        with sync:
            self.assertEqual(self.call().data[0].url, SIGNED_URL)
        with asynchronous:
            result = asyncio.run(aimage_generation(
                prompt="A mug", api_key="not-a-real-key", timeout=2,
            ))
        self.assertEqual(result.data[0].url, SIGNED_URL)
        self.assertEqual([r.method for r in self.requests], ["POST", "GET", "POST", "GET"])

    def test_standard_unsupported_parameters_reach_handler_and_fail_before_create(self):
        sync, _ = self.clients(lambda request: self.fail("Invalid parameters must not make a request"))
        with sync:
            for kwargs in ({"n": 2}, {"n": True}, {"size": "1024x1024"},
                           {"response_format": "b64_json"}, {"quality": "hd"},
                           {"style": "vivid"}, {"user": "someone"}, {"magic_hour_resolution": "4k"}):
                with self.subTest(kwargs=kwargs), self.assertRaises(Exception):
                    self.call(**kwargs)
        self.assertEqual(self.requests, [])

    def test_failure_after_acceptance_preserves_id_and_explicit_resume_never_posts(self):
        def handler(request):
            if request.method == "POST":
                return httpx.Response(200, json={"id": PROJECT_ID, "credits_charged": 5})
            raise httpx.ReadTimeout("private upstream details", request=request)
        sync, _ = self.clients(handler)
        with sync, self.assertRaises(Exception) as caught:
            self.call()
        self.assertIn(PROJECT_ID, str(caught.exception))
        self.assertNotIn("private upstream details", str(caught.exception))
        self.requests.clear()
        sync, _ = self.clients(lambda request: httpx.Response(200, json=complete()))
        with sync:
            result = self.call(magic_hour_project_id=PROJECT_ID)
        self.assertEqual(result.data[0].url, SIGNED_URL)
        self.assertEqual([r.method for r in self.requests], ["GET"])

    def test_ambiguous_paid_post_is_never_retried(self):
        def handler(request):
            raise httpx.ReadTimeout("possibly accepted", request=request)
        sync, _ = self.clients(handler)
        with sync, self.assertRaises(Exception) as caught:
            self.call()
        self.assertIn("acceptance is unknown", str(caught.exception))
        self.assertEqual([r.method for r in self.requests], ["POST"])

    def test_http_error_redirect_and_terminal_states_do_not_recreate(self):
        for response in (
            httpx.Response(401, json={"message": "private account info"}),
            httpx.Response(302, headers={"Location": "https://untrusted.example/"}),
            httpx.Response(200, json=complete(status="error")),
            httpx.Response(200, json=complete(status="canceled")),
            httpx.Response(200, json=complete(id="different-project")),
        ):
            with self.subTest(response=response):
                self.requests.clear()
                sync, _ = self.clients(lambda request: response)
                with sync, self.assertRaises(Exception) as caught:
                    self.call(magic_hour_project_id=PROJECT_ID)
                self.assertNotIn("private account info", str(caught.exception))
                self.assertEqual([r.method for r in self.requests], ["GET"])

    def test_polling_deadline_retains_project_id(self):
        clock = [0.0]
        handler = MagicHourImageProvider(deadline_seconds=1)
        def queued(request):
            clock[0] = 2.0
            return httpx.Response(200, json=complete(status="queued"))
        sync, _ = self.clients(queued)
        with sync, patch("magic_hour_litellm.time.monotonic", side_effect=lambda: clock[0]), self.assertRaises(MagicHourError) as caught:
            handler.image_generation(
                "z-image-turbo", "A mug", "not-a-real-key", None, None,
                {"magic_hour_options": {"magic_hour_project_id": PROJECT_ID}}, None,
            )
        self.assertEqual(caught.exception.status_code, 408)
        self.assertEqual(caught.exception.project_id, PROJECT_ID)
        self.assertTrue(self.requests)
        self.assertTrue(all(r.method == "GET" for r in self.requests))

    def test_custom_api_base_is_rejected_without_exposing_key(self):
        handler = MagicHourImageProvider()
        with self.assertRaises(MagicHourError) as caught:
            handler.image_generation("z-image-turbo", "A mug", "not-a-real-key", "https://elsewhere.example", None, {}, None)
        self.assertNotIn("not-a-real-key", str(caught.exception))

    def test_bare_custom_llm_call_fails_closed_before_create(self):
        sync, _ = self.clients(lambda request: self.fail("Bare dispatch must fail closed"))
        with sync, self.assertRaises(Exception) as caught:
            litellm.image_generation(
                model="magichour/z-image-turbo", prompt="A mug", api_key="not-a-real-key",
                n=2, num_retries=0,
            )
        self.assertIn("bare CustomLLM", str(caught.exception))
        self.assertEqual(self.requests, [])

    def test_proxy_hook_preserves_supported_options_and_rejects_count(self):
        hook = MagicHourProxyHook()
        async def run():
            data = await hook.async_pre_call_hook(None, None, {
                "model": "magic-hour-image", "prompt": "A mug", "n": 1,
                "size": "auto", "magic_hour_resolution": "1k",
            }, "image_generation")
            self.assertEqual(data["magic_hour_options"], {
                "n": 1, "size": "auto", "response_format": "url", "magic_hour_resolution": "1k",
            })
            self.assertNotIn("n", data)
            with self.assertRaises(Exception) as caught:
                await hook.async_pre_call_hook(None, None, {"model": "magic-hour-image", "n": 2}, "image_generation")
            self.assertEqual(caught.exception.status_code, 400)
            unchanged = {"model": "another-provider", "n": 2}
            self.assertIs(await hook.async_pre_call_hook(None, None, unchanged, "image_generation"), unchanged)
        asyncio.run(run())

    def test_proxy_default_options_survive_real_custom_dispatch(self):
        def handler(request):
            if request.method == "POST":
                self.assertEqual(json.loads(request.content)["resolution"], "640px")
                return httpx.Response(200, json={"id": PROJECT_ID, "credits_charged": 5})
            return httpx.Response(200, json=complete())
        _, asynchronous = self.clients(handler)
        async def run():
            data = await MagicHourProxyHook().async_pre_call_hook(
                None, None, {"model": "magic-hour-image", "prompt": "A mug"}, "image_generation",
            )
            self.assertEqual(data["magic_hour_options"], {
                "n": 1, "size": "auto", "response_format": "url", "magic_hour_resolution": "640px",
            })
            return await litellm.aimage_generation(
                model="magichour/z-image-turbo", prompt=data["prompt"],
                api_key="not-a-real-key", magic_hour_options=data["magic_hour_options"],
                num_retries=0, timeout=2,
            )
        with asynchronous:
            result = asyncio.run(run())
        self.assertEqual(result.data[0].url, SIGNED_URL)
        self.assertEqual([r.method for r in self.requests], ["POST", "GET"])


if __name__ == "__main__":
    unittest.main()
