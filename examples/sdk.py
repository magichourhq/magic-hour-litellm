"""One paid image create; call explicitly after setting MAGIC_HOUR_API_KEY."""
import os

from magic_hour_litellm import image_generation

result = image_generation(
    model="magichour/z-image-turbo",
    prompt="A studio product photograph of a blue ceramic mug on a warm white background",
    api_key=os.environ["MAGIC_HOUR_API_KEY"],
    n=1,
    size="auto",
    magic_hour_resolution="640px",
    timeout=600,
    num_retries=0,
)
print(result.data[0].provider_specific_fields)
print(result.data[0].url)
