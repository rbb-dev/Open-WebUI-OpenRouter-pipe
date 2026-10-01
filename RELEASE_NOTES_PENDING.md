# Release notes pending

## Behaviour changes

- **Config tab** — a save the database refuses to write now raises a durable banner instead of a toast alone.
  The banner names the fault, carries no Reload control, and leaves your staged edits in place, so nothing the
  refusal preserved can be discarded from the tab that preserved it. It clears on the next successful save or
  the next configuration load; today's toast is unchanged.
- **Auto-update** — the pause line now names *why* a release was paused, not just the code. The Auto-update
  row carries the version it stopped on and the refusal's own sentence, versions included; the
  `(apply manually or restart to re-arm)` hint is now shown only for a pause reason one of those can
  actually clear, and withheld where a manual apply re-runs the same guard.
- **frame reuse** — a frame the pipe extracts from a prior video is now bounded to 1920 on its **long edge**, not on
  its width alone. A portrait 4K phone video keeps its continuation frame instead of losing it with a ⚠️ note: a
  2160×3840 source now yields a 1080×1920 frame where it used to hand back a 1920×3414 one, past
  `VIDEO_FRAME_IMAGE_MAX_BYTES` at the shipped default, which dropped the frame with a note in the chat. Landscape
  sources and portrait sources at or under 1080×1920 are unchanged — the bound was already inside them.

  Detail quality on 9:16 sources is what this trades: 2160×3840 loses 3.2× its pixels, which is the point of the
  change and still leaves the anchor frame above 1080p-equivalent detail.

  Reach is common but content-dependent, not universal. A PNG's size follows entropy, not geometry, so the same
  shape keeps or loses its frame depending on the picture: busy, grainy, low-light or fine-detail 4K portrait clips
  lose it, clean or smooth ones keep it. What made it worth fixing is that the boundary is a knife edge — one stop
  brighter or one notch sharper flips it — and nothing beyond a ⚠️ line said so.

- **frame reuse, memory** — with the long-edge bound in place, the extractor asks no decoder for a frame larger than
  1920 on either axis, whatever the source, so its per-frame footprint is now bounded at about 3.7 Mpx and 10.6 MiB
  instead of scaling with the source's short side. The pixel-cap gate that used to refuse an over-cap portrait source
  from its header before decoding is no longer reachable — the bound keeps the geometry inside the cap — so a tall
  over-cap source such as 1920×20000 is now decoded rather than refused from a header read, and the pipe pays that
  decode. It already paid it for every wide over-cap source (10000×3000 is the documented rescue), so what is given
  up is only the tall-and-narrow half of an asymmetry with no principled basis. The gate and the decoded-size check
  behind it stay in the code for a header that lies about the source.

- **fusion** — a Fusion panel member's tool file is no longer filed against the outer chat. The file is still
  stored and still rendered in the panel, but it no longer appears in the chat's file list: the tool executor
  re-checks `fusion_inner` before it hands `chat_id`/`message_id` to Open WebUI's upload, so a member's file
  cannot produce a `chat_file` row naming a real chat in a message that does not exist.
- **image contracts** — repointing `BASE_URL` now drops the shared published image contracts immediately, and not
  only when a contract sweep happens to run. The drop sat below the sweep's staleness gate, so a pass with nothing
  stale to do returned before it, and a gateway admin running with the four image filter valves off kept being
  served the previous gateway's aspect-ratio, resolution and seed limits until some later sweep ran. The identity
  is now adopted before the gate, so one stored value (`API_KEY`) has one reader and one answer.
- **API key** — a stored `API_KEY` the key gate refuses now produces the authentication card on the streaming and the
  housekeeping-task legs too, instead of a 401 or an opaque "Unexpected error in streaming loop". Both of those legs
  read the stored field with a bare decrypt instead of through `Pipe._resolve_openrouter_api_key`, so an encrypted
  value that cannot be decrypted went out as an empty `Bearer ` and an encrypted non-`sk-` value the gate refuses went
  out working; a stored value with padding was sent with its padding. Both legs now go through the gate, so one
  misconfiguration has one answer on every leg.
