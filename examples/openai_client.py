"""One paid image through your locally running, authenticated LiteLLM proxy."""
import os

from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:4000/v1",
    api_key=os.environ["LITELLM_MASTER_KEY"],
    max_retries=0,
    timeout=650,
)
result = client.images.generate(
    model="magic-hour-image",
    prompt="A studio product photograph of a blue ceramic mug on a warm white background",
    n=1,
    size="auto",
    response_format="url",
    extra_body={"magic_hour_resolution": "640px"},
)
print(result.data[0].url)
