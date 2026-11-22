# Release notes pending

## Behaviour changes

- **Image sizes chosen for Gemini image chat models are used again** - Since 2.8.0 the size chosen in the image settings of a chat model that also draws (Nano Banana Pro, Nano Banana Pro Preview, Nano Banana 2 and its Preview, Nano Banana 2.1) went to OpenRouter under the name its image endpoint uses, `resolution`. On chat requests OpenRouter reads that setting only as `image_size` and accepts the other name without acting on it, so every image came back at the model's default 1K size, whatever was chosen. The pipe now sends the size as `image_size` on chat requests; image-only models, which use OpenRouter's image endpoint, keep `resolution`. OpenRouter lists 4K for Nano Banana Pro (`google/gemini-3-pro-image`) but refuses it when asked, naming the models that can make 4K, such as Nano Banana Pro Preview; that refusal is now shown instead of a 1K image.
