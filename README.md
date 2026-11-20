# Magic Hour for LiteLLM

Generate images with [Magic Hour](https://magichour.ai) through [LiteLLM](https://docs.litellm.ai), from the SDK or the OpenAI-compatible proxy. It's a LiteLLM custom provider, not one built into LiteLLM.

It supports one model, `z-image-turbo`, and one square image per call.

## Install

```sh
pip install magic-hour-litellm
```

Requires Python 3.10 or newer and LiteLLM 1.81.7 or newer. Set `MAGIC_HOUR_API_KEY` to a key from your [Magic Hour developer dashboard](https://magichour.ai/developer).

## SDK

```python
from magic_hour_litellm import image_generation

image = image_generation(
    model="magichour/z-image-turbo",
    prompt="A blue ceramic mug on a warm white background",
    magic_hour_resolution="1k",
)
print(image.data[0].url)
print(image.data[0].provider_specific_fields)  # project_id, credits_charged, expires_at
```

`await aimage_generation(...)` takes the same arguments. Use these functions rather than `litellm.image_generation`: LiteLLM drops image options before calling a custom provider, and these pass them through. A plain `litellm.image_generation` call fails before any request.

| Argument | Values |
| --- | --- |
| `model` | `magichour/z-image-turbo` |
| `prompt` | Non-empty text |
| `magic_hour_resolution` | `640px` (default), `1k`, `2k`: the longest edge, not exact dimensions |
| `magic_hour_project_id` | An accepted project to resume, without creating a new one |
| `timeout` | Seconds to wait, up to 600 (the default) |
| `n`, `size`, `response_format` | Only `1`, `auto` and `url` |

Anything else fails before a request is sent.

## Proxy

```sh
pip install magic-hour-litellm 'litellm[proxy]' prisma
export LITELLM_MASTER_KEY='<a private proxy key>'
litellm --config examples/proxy.yaml --host 127.0.0.1 --port 4000
```

[examples/proxy.yaml](examples/proxy.yaml) registers the provider and the `magic_hour_litellm.proxy_hook` callback, which passes the image options through. Keep the hook enabled. Without it, requests fail rather than lose their options. The hook covers the `magic-hour-image` model name. For another name, configure `MagicHourProxyHook(model_names=(...))` in your own config module.

LiteLLM's proxy imports Prisma when it rejects a request, even with no database, which is why it's in the install line.

Call it with any OpenAI client. See [examples/openai_client.py](examples/openai_client.py).

## Credits and failures

Each new image is a paid Magic Hour job. The provider sends the create request once, then only polls that project.

- **Don't retry creates.** The SDK functions refuse `num_retries` other than 0, and the example proxy config sets `num_retries: 0`. Don't add fallbacks or retry middleware for this model.
- **After acceptance, errors name the project.** Resume with `magic_hour_project_id` instead of generating again. A timeout doesn't cancel the job.
- **If a create fails before returning an ID,** the job may still have been accepted. Check your Magic Hour projects before creating again.
- **Download URLs expire.** Resuming the same project returns a fresh one.

Errors don't include prompts, keys, signed URLs or response bodies. LiteLLM's own logging may, so configure it for your setup.

## Development

```sh
pip install -e . 'litellm[proxy]'
python -m unittest discover -s tests
```

The tests replace HTTP with a fake, so they spend no credits.

## License

MIT
