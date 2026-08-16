
from __future__ import annotations

IMAGE_FIELD_ROUTES: dict[str, str] = {
    "aspect_ratio": "per-model control, drawn when the model publishes the shapes it takes",
    "background": "per-model control, drawn when the model publishes its choices",
    "input_references": (
        "the images attached to the turn, narrowed by the reference-images control, plus "
        "any links given in the reference-links control"
    ),
    "model": "the model picked in the chat",
    "n": "per-model control, bounded by the published range",
    "output_compression": "per-model control, bounded by the published range",
    "output_format": "per-model control, drawn when the model publishes its containers",
    "prompt": "the message typed in the chat, with the model's own system text prefixed",
    "provider": (
        "the routing picker for the fields this request format accepts, and the "
        "provider-options control for anything addressed to one company"
    ),
    "quality": "per-model control, drawn when the model publishes its tiers",
    "resolution": "per-model control, drawn when the model publishes its tiers",
    "seed": "per-model control, drawn when the model declares it supports one",
    "size": "control drawn on every model, since no model's contract describes this field",
    "stream": (
        "set for you, whenever every endpoint that could serve the request publishes "
        "native streaming, so the chat shows progress while the picture is drawn"
    ),
}

IMAGE_FIELD_GAPS: dict[str, str] = {}

VIDEO_FIELD_ROUTES: dict[str, str] = {
    "aspect_ratio": "per-model control, drawn from the shapes the catalogue lists",
    "duration": "per-model control, drawn from the lengths the catalogue lists",
    "frame_images": "the images attached to the turn, narrowed by the frames control",
    "generate_audio": "per-model control, drawn when the catalogue declares the toggle",
    "model": "the model picked in the chat",
    "prompt": "the message typed in the chat",
    "provider": (
        "the typed per-model controls and the provider-options control, both of which "
        "address the company serving the request"
    ),
    "resolution": "per-model control, drawn from the tiers the catalogue lists",
    "seed": "per-model control, drawn when the catalogue declares one",
    "size": "per-model control, drawn from the sizes the catalogue lists",
}

VIDEO_FIELD_GAPS: dict[str, str] = {
    "callback_url": (
        "OpenRouter would call this address when the video is done. An Open WebUI "
        "function has no address of its own to be called back on, and the secret that "
        "proves such a call genuine is set on an OpenRouter workspace rather than sent "
        "with a request, so nothing here could tell a real callback from a forged one. "
        "The pipe polls for the result instead, which needs neither."
    ),
    "input_references": (
        "chosen for you: the classifier that reads the chat decides when an earlier "
        "frame or an attachment becomes a reference, and there is no control to set the "
        "list directly or to supply a link. Audio and video references, which this field "
        "also accepts, are unreachable for the same reason."
    ),
}

IMAGE_REQUEST_FIELDS: frozenset[str] = frozenset(IMAGE_FIELD_ROUTES) | frozenset(
    IMAGE_FIELD_GAPS
)
VIDEO_REQUEST_FIELDS: frozenset[str] = frozenset(VIDEO_FIELD_ROUTES) | frozenset(
    VIDEO_FIELD_GAPS
)

assert not (set(IMAGE_FIELD_ROUTES) & set(IMAGE_FIELD_GAPS)), (
    "an image request field is either reached or it is not; listing one as both leaves "
    "the reader no way to tell which"
)
assert not (set(VIDEO_FIELD_ROUTES) & set(VIDEO_FIELD_GAPS)), (
    "a video request field is either reached or it is not; listing one as both leaves "
    "the reader no way to tell which"
)
assert all(
    reason.strip()
    for reason in (*IMAGE_FIELD_ROUTES.values(), *IMAGE_FIELD_GAPS.values())
), "an image request field listed with no reason records nothing"
assert all(
    reason.strip()
    for reason in (*VIDEO_FIELD_ROUTES.values(), *VIDEO_FIELD_GAPS.values())
), "a video request field listed with no reason records nothing"
