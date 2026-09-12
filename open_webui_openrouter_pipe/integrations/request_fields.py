
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
    "input_references": (
        "chosen for you: the classifier that reads the chat decides when an earlier "
        "frame or an attachment becomes a reference, and there is no control to set the "
        "list directly or to supply a link. Audio and video references, which this field "
        "also accepts, are reachable only the same way."
    ),
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
}

VIDEO_CATALOG_FIELD_ROUTES: dict[str, str] = {
    "allowed_passthrough_parameters": "the names a typed control is drawn for, or a free-text box when none exists",
    "description": "the capability line of the model's help card",
    "generate_audio": "the audio toggle, drawn when the catalogue does not declare it off",
    "id": "the model's identity everywhere: filter name, marker, catalogue lookup, help card",
    "name": "the display name on the filter and at the head of the help card",
    "seed": "the seed control, drawn when the catalogue does not declare it off",
    "supported_aspect_ratios": "the aspect-ratio control, drawn from the shapes listed",
    "supported_durations": "the duration control, drawn from the lengths listed",
    "supported_frame_images": "the frames control, and the frame-controls line of the help card",
    "supported_resolutions": "the resolution control, drawn from the tiers listed",
    "supported_sizes": "the size control, drawn from the exact dimensions listed",
}

VIDEO_CATALOG_FIELD_GAPS: dict[str, str] = {
    "canonical_slug": (
        "the dated spelling of the same model, such as runway/aleph-2-20260729. Every "
        "lookup in this pipe keys on the undated id the chat header shows, so a second "
        "spelling would only give two names for one model and a way for them to disagree."
    ),
    "created": (
        "the timestamp the model was published. Nothing here orders or filters models by "
        "age: the catalogue arrives in OpenRouter's own order and is shown in it, so the "
        "date would be read by nobody and would have to be kept true by somebody."
    ),
    "creativity": (
        "the model publishes the two modes it accepts, but OpenRouter's video request "
        "format documents no field that carries them and the model names only "
        "safety_tolerance as a passthrough parameter, so there is no route to send one. "
        "Returned to the panel rather than guessed at; see round 21 in the ledger."
    ),
    "hugging_face_id": (
        "null on every video model OpenRouter publishes, so there is nothing to read. If "
        "one ever carries a value, this entry is wrong and the census test will not catch "
        "it, because a gap with a reason is exactly what it is asked to accept."
    ),
    "pricing_skus": (
        "deliberately never read. A rate shown to a reader goes stale in silence whoever "
        "supplied it, so no card, filter or page in this pipe quotes one; what a "
        "generation actually cost is reported after the poll instead. Pinned by "
        "test_no_video_price_reaches_a_user_at_all."
    ),
    "upscale_factor": (
        "the model publishes the range it accepts, but OpenRouter's video request format "
        "documents no field that carries it and the model names only safety_tolerance as "
        "a passthrough parameter, so there is no route to send one. Returned to the panel "
        "rather than guessed at; see round 21 in the ledger."
    ),
}

VIDEO_CATALOG_FIELDS: frozenset[str] = frozenset(VIDEO_CATALOG_FIELD_ROUTES) | frozenset(
    VIDEO_CATALOG_FIELD_GAPS
)

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
