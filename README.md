# Magic Hour for LiteLLM

A first-party, installable **CustomLLM image adapter** for LiteLLM 1.106.3. It
connects LiteLLM's image-generation SDK and OpenAI-compatible proxy to Magic Hour.
It is not a built-in LiteLLM provider or an upstream marketplace listing.

The initial supported operation is **one `z-image-turbo` image**. A call creates
one Magic Hour project, polls that same project, and returns its signed download
URL in a normal `ImageResponse`. It does not download or transform the image.

## Verified release status

The native SDK release check passed with LiteLLM 1.106.3: one create request,
three same-project status reads, and an actual charge of five Magic Hour credits.
The returned PNG was downloaded, fully decoded and visually checked: a usable
blue ceramic mug, 640 × 640 pixels, 439,818 bytes. Its SHA256 is
`c5f3f72a60eac48bfa146b8d196de1c34d9b195131990a75c81d13fdc11115e6`.
This is one verified result, not a guarantee of price or exact dimensions for
every request.

All 12 mocked tests passed, including the proxy hook. The built 0.1.0 wheel
was installed in a clean target and exercised through the stock LiteLLM proxy
and normal OpenAI client. The proxy rejected unauthenticated requests with 401,
exposed the configured model to an authenticated client, and recovered that
already-completed image with its project ID, credit metadata and a fresh HTTPS
download URL. This proxy check requested no new generation. The owned server
and listener were closed afterward. Native paid creation was verified through
the SDK, while native proxy verification used recovery only.

This is a first-party custom adapter, not built-in LiteLLM support or upstream
acceptance. No PyPI publication is implied by these source installation steps.

## Install from source

From the root of this source checkout, use Python 3.10 or later in your
application's own virtual environment:

```sh
python -m pip install .
export MAGIC_HOUR_API_KEY='<your Magic Hour API key>'
```

Obtain a key from your [Magic Hour developer dashboard](https://magichour.ai/developer).
Image generation spends Magic Hour credits. Availability, account eligibility,
resolution and actual charges are controlled by Magic Hour. This adapter never
converts credits into an invented dollar price or reports a zero-cost generation.

## LiteLLM SDK

```python
import os
from magic_hour_litellm import image_generation

image = image_generation(
    model="magichour/z-image-turbo",
    prompt="A studio product photograph of a blue ceramic mug on a warm white background",
    api_key=os.environ["MAGIC_HOUR_API_KEY"],
    n=1,
    size="auto",
    response_format="url",
    magic_hour_resolution="640px",
    timeout=600,
    num_retries=0,
)
print(image.data[0].url)
print(image.data[0].provider_specific_fields)
```

The asynchronous `await magic_hour_litellm.aimage_generation(...)` interface
accepts the same arguments. These thin entrypoints validate the raw image options,
disable automatic retries and invoke LiteLLM's real CustomLLM SDK dispatch.
Registration preserves other custom providers and replaces only `magichour`.

Supported inputs:

| Parameter | Supported value |
| --- | --- |
| model | `magichour/z-image-turbo` |
| prompt | Nonempty text, sent unchanged to Magic Hour |
| n | `1`, or omitted |
| size | `auto`, or omitted |
| response_format | `url`, or omitted |
| magic_hour_resolution | `640px` (default), `1k`, `2k` |
| magic_hour_project_id | An existing project ID, to resume without creating |
| timeout | Positive numeric polling budget, capped at 600 seconds |

Magic Hour defines resolution as a **maximum longest edge**, not a guarantee of
exact pixel dimensions. The adapter requests square aspect ratio `1:1`; it
rejects sizes such as `1024x1024` rather than promise an exact size. Unsupported
quality, style, user, count, response format and extra parameters are rejected
before any API call. Prompt text is not interpreted or rewritten.

The successful image's `provider_specific_fields` includes `project_id`, the
download's `expires_at`, and the latest `credits_charged` when returned by Magic
Hour. Credits can change after a refund. Keep download URLs private and use them
before expiry; a same-ID retrieval can refresh an expired URL.

## OpenAI-compatible LiteLLM proxy

Install LiteLLM's proxy extra and its auth-error dependency into the same
environment, then start it on your
loopback interface. Keep the key in the proxy environment, never in browser code:

```sh
python -m pip install 'litellm[proxy]==1.106.3' 'prisma==0.15.0'
export LITELLM_MASTER_KEY='<a private proxy key>'
litellm --config examples/proxy.yaml --host 127.0.0.1 --port 4000
```

LiteLLM 1.106.3 imports Prisma in its authentication-error handler even without
a database. The explicit package above keeps unauthenticated requests on the
normal rejection path; no database, generated client or Prisma engine is needed
for this configuration. The broader `extra-proxy` extra is not required here.

`examples/openai_client.py` uses the normal OpenAI client against that proxy with
`max_retries=0`. Run it explicitly when ready to spend credits. The supplied
proxy config sets `num_retries: 0`; do not add fallback models or automatic
retry middleware to this paid workflow.

The adapter uses the official `custom_provider_map` loader and
`CustomLogger.async_pre_call_hook`. LiteLLM 1.106.3 drops standard image parameters
before calling a plain CustomLLM handler, so the supplied SDK entrypoints and
proxy hook validate and preserve them in a provider-specific option envelope.
Bare `litellm.image_generation` calls without that envelope fail before making
an API call. The supplied proxy hook is scoped to its `magic-hour-image` model
alias; if you rename the alias, configure `MagicHourProxyHook(model_names=(...))`
in your own importable configuration module. Keep the hook enabled. Missing it
fails closed rather than silently ignoring image count or size. No LiteLLM source
monkeypatch, compatibility-provider flag or custom network bridge is used.

## Failure and recovery

The adapter's HTTP client does not retry the create request or follow redirects.
It sends a maximum of one `POST /v1/ai-image-generator` per invocation, then only
`GET /v1/image-projects/{id}`. A timeout does not cancel the Magic Hour job.
The polling budget is checked between requests and sleeps. HTTPX applies the
remaining budget as a per-phase network timeout; this is not guaranteed hard
wall-clock cancellation of an in-flight request.

After acceptance, handled request failures retain the project ID in error text.
Until a result or handled error reaches the caller, that ID is only in process
memory: a hard crash or cancellation can prevent its delivery. If you have no
ID, inspect your Magic Hour projects before deciding whether to create again.
To recover **the same paid job**, call with a nonempty prompt and
`magic_hour_project_id` set:

```python
recovered = image_generation(
    model="magichour/z-image-turbo",
    prompt="Recovery of an existing image project",  # required by LiteLLM; not sent on resume
    magic_hour_project_id="<accepted-project-id>",
    api_key=os.environ["MAGIC_HOUR_API_KEY"],
    num_retries=0,
    timeout=600,
)
```

Resume performs no POST. If the connection fails before a create response yields
an ID, acceptance is unknown: inspect your Magic Hour projects before deciding
whether to create again. There is no public create idempotency contract here.
Caller retries, proxy retries, process restarts and repeated button presses can
create duplicate paid projects; this adapter cannot guarantee exactly-once work
under those conditions. Do not automatically rerun a failed create.

Ordinary SDK/proxy logging may include prompts and signed output URLs. Use a
private application/proxy, configure its access and logging appropriately, and
never publish API keys or raw request/response logs. The adapter's own error text
does not echo provider response bodies, prompts, keys or signed URLs.

## Tests and contribution

The SDK-only installation does not include FastAPI. The complete unit suite
exercises the proxy hook's normal HTTP errors and needs the same FastAPI version
range declared by LiteLLM 1.106.3's proxy extra:

```sh
python -m pip install 'fastapi>=0.136.3,<1.0'
python -m unittest discover -s tests -v
```

This minimal unit-test dependency does not install or verify the complete proxy
runtime. To run the actual server, use the `litellm[proxy]` installation above.

The tests use mocked HTTP transport and actual LiteLLM sync/async dispatch. They
cover parameter preservation/rejection, one create followed by same-ID polling,
unchanged signed URLs, actual credit metadata, same-ID resume, ambiguous create
failure, redirects, authorization failures, terminal states and polling timeout.
They perform no real generation. Native SDK/proxy and media verification are
separate release checks; a passing unit suite alone does not prove them.

The supported first-party installation route works independently of upstream
acceptance. An upstream built-in-provider proposal would be a separate change
to LiteLLM's routing, image config, docs and tests, following its
[contribution guidelines](https://docs.litellm.ai/docs/extras/contributing_code)
and any required contributor agreement. No upstream inclusion is implied.

API contract:
[Magic Hour image generation](https://docs.magichour.ai/api-reference/ai-image-generator/generate-ai-images),
[image project status](https://docs.magichour.ai/api-reference/image-projects/get-image-details),
[LiteLLM CustomLLM](https://docs.litellm.ai/docs/providers/custom_llm_server).
