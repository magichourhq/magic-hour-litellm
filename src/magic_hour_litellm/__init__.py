"""A first-party Magic Hour CustomLLM image handler, tested with LiteLLM 1.106.3."""

import asyncio
import math
import os
import re
import time

import httpx
import litellm
from litellm.llms.custom_llm import CustomLLM, CustomLLMError
from litellm.integrations.custom_logger import CustomLogger
from litellm.types.utils import ImageObject, ImageResponse

API_BASE = "https://api.magichour.ai"


class MagicHourError(CustomLLMError):
    """Safe error text retains a known accepted project ID for explicit recovery."""

    def __init__(self, status_code, message, project_id=None):
        self.project_id = project_id
        if project_id:
            message += (
                f" Project ID: {project_id}. Resume with "
                f"magic_hour_project_id='{project_id}'; do not create another project."
            )
        super().__init__(status_code=status_code, message=message)


class MagicHourImageProvider(CustomLLM):
    """One z-image-turbo image per call; bounded polling and explicit same-ID resume."""

    def __init__(self, *, poll_interval=2.0, deadline_seconds=600.0):
        super().__init__()
        if not math.isfinite(poll_interval) or poll_interval <= 0:
            raise ValueError("poll_interval must be finite and positive")
        if not math.isfinite(deadline_seconds) or deadline_seconds <= 0:
            raise ValueError("deadline_seconds must be finite and positive")
        self.poll_interval = poll_interval
        self.deadline_seconds = deadline_seconds

    @staticmethod
    def _options(params):
        params = {key: value for key, value in params.items() if value is not None}
        allowed = {"n", "size", "response_format", "magic_hour_resolution", "magic_hour_project_id"}
        if params.keys() - allowed:
            raise MagicHourError(400, "Unsupported image parameter(s): " + ", ".join(sorted(params.keys() - allowed)))
        if type(params.get("n", 1)) is not int or params.get("n", 1) != 1:
            raise MagicHourError(400, "Only n=1 is supported.")
        if params.get("size", "auto") != "auto":
            raise MagicHourError(400, "Exact pixel sizes are unsupported; omit size or use auto.")
        if params.get("response_format", "url") != "url":
            raise MagicHourError(400, "Only response_format='url' is supported.")
        resolution = params.get("magic_hour_resolution", "640px")
        if resolution not in ("640px", "1k", "2k"):
            raise MagicHourError(400, "Supported maximum-edge resolutions: 640px, 1k, 2k.")
        project_id = params.get("magic_hour_project_id")
        if project_id is not None and (
            not isinstance(project_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", project_id)
        ):
            raise MagicHourError(400, "Invalid Magic Hour project ID.")
        return resolution, project_id

    def _prepare(self, model, prompt, api_key, api_base, optional_params, timeout):
        if model != "z-image-turbo":
            raise MagicHourError(400, "Supported model: magichour/z-image-turbo.")
        if not isinstance(prompt, str) or not prompt.strip():
            raise MagicHourError(400, "A nonempty prompt is required.")
        if api_base not in (None, API_BASE, API_BASE + "/"):
            raise MagicHourError(400, "Custom API endpoints are unsupported.")
        params = dict(optional_params or {})
        options = params.pop("magic_hour_options", None)
        if params or not isinstance(options, dict):
            raise MagicHourError(400, "Use magic_hour_litellm.image_generation or the supplied proxy hook; bare CustomLLM image calls drop standard options in LiteLLM 1.106.3.")
        resolution, project_id = self._options(options)
        key = api_key or os.environ.get("MAGIC_HOUR_API_KEY")
        if not isinstance(key, str) or not key:
            raise MagicHourError(401, "Set MAGIC_HOUR_API_KEY or pass api_key.")
        # Check the polling budget between requests; HTTPX network timeouts
        # apply per phase, so this is not a hard wall-clock cancellation.
        if timeout is None:
            budget = self.deadline_seconds
        elif isinstance(timeout, (int, float)) and not isinstance(timeout, bool):
            budget = min(float(timeout), self.deadline_seconds)
        else:
            raise MagicHourError(400, "timeout must be a positive number of seconds.")
        if not math.isfinite(budget) or budget <= 0:
            raise MagicHourError(400, "timeout must be a finite positive number of seconds.")
        body = {
            "image_count": 1,
            "model": "z-image-turbo",
            "aspect_ratio": "1:1",
            "resolution": resolution,
            "style": {"prompt": prompt, "tool": "general"},
        }
        return key, body, project_id, time.monotonic() + budget

    @staticmethod
    def _remaining(deadline, project_id):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise MagicHourError(408, "Magic Hour polling deadline exceeded.", project_id)
        return remaining

    @staticmethod
    def _json(response, project_id=None):
        if not response.is_success:
            raise MagicHourError(response.status_code, f"Magic Hour returned HTTP {response.status_code}.", project_id)
        try:
            payload = response.json()
        except ValueError:
            raise MagicHourError(502, "Magic Hour returned invalid JSON.", project_id) from None
        if not isinstance(payload, dict):
            raise MagicHourError(502, "Magic Hour returned an invalid response.", project_id)
        return payload

    @staticmethod
    def _accepted(payload):
        project_id = payload.get("id")
        if not isinstance(project_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", project_id):
            raise MagicHourError(502, "Create response had no usable project ID. Acceptance is unknown; do not retry automatically.")
        return project_id

    @staticmethod
    def _result(payload, project_id):
        if payload.get("id") != project_id:
            raise MagicHourError(502, "Magic Hour returned a different project ID.", project_id)
        status = payload.get("status")
        if status in ("error", "canceled", "draft"):
            raise MagicHourError(422, f"Magic Hour project status: {status}.", project_id)
        if status in ("queued", "rendering"):
            return None
        if status != "complete":
            raise MagicHourError(502, "Magic Hour returned an unknown project status.", project_id)
        downloads = payload.get("downloads")
        if not isinstance(downloads, list) or len(downloads) != 1:
            raise MagicHourError(502, "Expected exactly one completed image download.", project_id)
        download = downloads[0]
        if not isinstance(download, dict) or not isinstance(download.get("url"), str) or not download["url"].startswith("https://"):
            raise MagicHourError(502, "Magic Hour returned an invalid image download.", project_id)
        metadata = {"project_id": project_id, "expires_at": download.get("expires_at")}
        credits = payload.get("credits_charged")
        if type(credits) is int:
            metadata["credits_charged"] = credits
        return ImageResponse(
            created=int(time.time()),
            data=[ImageObject(url=download["url"], provider_specific_fields=metadata)],
        )

    def image_generation(self, model, prompt, api_key, api_base, model_response,
                         optional_params, logging_obj, timeout=None, client=None):
        key, body, project_id, deadline = self._prepare(model, prompt, api_key, api_base, optional_params, timeout)
        try:
            # httpx defaults to zero transport retries; never enable redirects
            # or retry the paid POST, even when its acceptance is ambiguous.
            with httpx.Client(base_url=API_BASE, headers={"Authorization": f"Bearer {key}"}, follow_redirects=False) as http:
                if project_id is None:
                    response = http.post("/v1/ai-image-generator", json=body, timeout=self._remaining(deadline, None))
                    project_id = self._accepted(self._json(response))
                while True:
                    response = http.get(f"/v1/image-projects/{project_id}", timeout=self._remaining(deadline, project_id))
                    result = self._result(self._json(response, project_id), project_id)
                    if result is not None:
                        return result
                    time.sleep(min(self.poll_interval, self._remaining(deadline, project_id)))
        except httpx.HTTPError:
            message = "Magic Hour transport failed."
            if project_id is None:
                message += " Create acceptance is unknown; inspect your projects before retrying."
            raise MagicHourError(502, message, project_id) from None

    async def aimage_generation(self, model, prompt, model_response, api_key, api_base,
                               optional_params, logging_obj, timeout=None, client=None):
        key, body, project_id, deadline = self._prepare(model, prompt, api_key, api_base, optional_params, timeout)
        try:
            async with httpx.AsyncClient(base_url=API_BASE, headers={"Authorization": f"Bearer {key}"}, follow_redirects=False) as http:
                if project_id is None:
                    response = await http.post("/v1/ai-image-generator", json=body, timeout=self._remaining(deadline, None))
                    project_id = self._accepted(self._json(response))
                while True:
                    response = await http.get(f"/v1/image-projects/{project_id}", timeout=self._remaining(deadline, project_id))
                    result = self._result(self._json(response, project_id), project_id)
                    if result is not None:
                        return result
                    await asyncio.sleep(min(self.poll_interval, self._remaining(deadline, project_id)))
        except httpx.HTTPError:
            message = "Magic Hour transport failed."
            if project_id is None:
                message += " Create acceptance is unknown; inspect your projects before retrying."
            raise MagicHourError(502, message, project_id) from None


provider = MagicHourImageProvider()


def register():
    """Register this handler without replacing other configured custom providers."""
    entries = [entry for entry in litellm.custom_provider_map if entry["provider"] != "magichour"]
    litellm.custom_provider_map = entries + [{"provider": "magichour", "custom_handler": provider}]


def _sdk_call(prompt, model, timeout, api_key, options):
    if model != "magichour/z-image-turbo":
        raise MagicHourError(400, "Supported model: magichour/z-image-turbo.")
    options = {"n": 1, "size": "auto", "response_format": "url",
               "magic_hour_resolution": "640px", **options}
    provider._options(options)
    register()
    return dict(model=model, prompt=prompt, api_key=api_key, timeout=timeout,
                num_retries=0, magic_hour_options=options)


def image_generation(*, prompt, model="magichour/z-image-turbo", timeout=600, api_key=None, num_retries=0, **options):
    """Validate raw options before LiteLLM's official CustomLLM SDK dispatch."""
    if num_retries != 0:
        raise MagicHourError(400, "Automatic create retries are unsupported; num_retries must be zero.")
    return litellm.image_generation(**_sdk_call(prompt, model, timeout, api_key, options))


async def aimage_generation(*, prompt, model="magichour/z-image-turbo", timeout=600, api_key=None, num_retries=0, **options):
    """Asynchronous version of image_generation, with automatic retries disabled."""
    if num_retries != 0:
        raise MagicHourError(400, "Automatic create retries are unsupported; num_retries must be zero.")
    return await litellm.aimage_generation(**_sdk_call(prompt, model, timeout, api_key, options))


class MagicHourProxyHook(CustomLogger):
    """Preserve and validate raw image options using LiteLLM's official proxy hook."""

    def __init__(self, model_names=("magic-hour-image",)):
        super().__init__()
        self.model_names = frozenset(model_names)

    async def async_pre_call_hook(self, user_api_key_dict, cache, data, call_type):
        if call_type != "image_generation" or data.get("model") not in self.model_names:
            return data
        # FastAPI is installed by the proxy extra, not required for SDK use.
        from fastapi import HTTPException
        if "magic_hour_options" in data:
            raise HTTPException(400, "magic_hour_options is reserved for the Magic Hour proxy hook.")
        keys = ("n", "size", "response_format", "quality", "style", "user",
                "imageConfig", "magic_hour_resolution", "magic_hour_project_id")
        options = {"n": 1, "size": "auto", "response_format": "url",
                   "magic_hour_resolution": "640px",
                   **{key: data[key] for key in keys if key in data}}
        try:
            provider._options(options)
        except MagicHourError as error:
            raise HTTPException(error.status_code, error.message) from None
        for key in keys:
            data.pop(key, None)
        data["magic_hour_options"] = options
        return data


proxy_hook = MagicHourProxyHook()
