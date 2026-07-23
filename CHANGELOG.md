# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
Each entry links to the commit that introduced it.

## [2.7.3](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/releases/tag/v2.7.3) — 2026-07-23

### 🚀 Features

- **filters** — enable provider routing filters by default in new chats ([#57](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/issues/57)) ([`ee2f85e`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/ee2f85eb0b60a4eab3f669d16787aa5115a9ee50))

### 🐛 Bug Fixes

- **filters** — list every provider in provider routing dropdowns via the model endpoints API ([#57](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/issues/57)) ([`1c8cbc9`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/1c8cbc9a15dc6c0729c99b0ebc6c91bfe120cd41))
- **catalog** — fetch the frontend catalog when only description sync is enabled ([`752f1fc`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/752f1fcd8c905db6e54a8695069f011911987652))
## [2.7.2](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/releases/tag/v2.7.2) — 2026-07-22

### 🚀 Features

- **config** — choose guid, email, or name for the OpenRouter user id ([`184d99c`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/184d99c4c4f6c78f65c8bd9a1f5fee30541dad38))

### 🐛 Bug Fixes

- **logging** — let the session log assembler exit cleanly on shutdown ([`5a1ea16`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/5a1ea16fc0dee07512880994b431af5f0e6e3bd7))

### 📚 Documentation

- regenerate changelog for v2.7.2 ([`ae34e71`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/ae34e71800a9eaedbdf0399c9e16e4b5e607dd88))
## [2.7.1](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/releases/tag/v2.7.1) — 2026-07-22

### 🐛 Bug Fixes

- **dashboard** — let the embed iframe shrink when switching to a shorter tab ([`86c8dea`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/86c8dea3359229829ef264eec4c869232d7b352c))
- **fusion** — drop live-object deep copies and sanitize member failures ([`7a5596c`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/7a5596cb07abc40c3250044d2ff9ca4e66644325))
- **tools** — keep $ref and $defs intact when strictifying MCP schemas ([`de190bb`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/de190bbf255a12985d809e52a0d6be8d475ea05e))
- **tools** — stop retrying and cross-tripping side-effectful MCP tools ([`920b1c2`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/920b1c2bf1f337c2726619bda376bfbf964ddc16))
- **fusion** — also emit the final answer as a native output item ([`7198a43`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/7198a439d25d24f51b803d90ed99fdb5782a13d0))
- **fusion** — dedupe inner tools and capture MCP media links ([`afdfdce`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/afdfdcedab8951965e2515d9d9ded50aa3d41b57))
- **tools** — keep original tool params and guard persistence drops ([`9f0dc4f`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/9f0dc4fbbd4c71bc9fbf9cfa43b8149ab6809935))
- **models** — admit variant suffixes of ZDR-capable bases under enforcement ([`fc55cb8`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/fc55cb8fa2c15cd1d126acad2b12c9203f139fc0))
- **tools** — harvest real URL citations from MCP and custom tool results ([`34aa475`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/34aa475b254fd9922d41e283e538257a717ca2f3))

### 📚 Documentation

- **dashboard** — replace dashboard screenshots with a walkthrough gif ([`5b31e87`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/5b31e874a8c826b24c038f2a9bfc2fa4d25fcf73))
- **readme** — stop constraining the dashboard gif to 49% width ([`eb6eadf`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/eb6eadf55feccf4613dd0e6bf32bd59483c32826))
- regenerate changelog for v2.7.1 ([`376d578`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/376d578a2252711bc3d97b68cec360d5596aa587))
## [2.7.0](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/releases/tag/v2.7.0) — 2026-07-20

### 🚀 Features

- plugin system with dynamic discovery, hooks, and OWUI seam guard ([`f1fc116`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/f1fc116d81bc5a2f60d85acee74b539a582062c3))
- pipe_dashboard plugin with live dashboard, usage stats, and config tab ([`8777659`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/877765983ab42376bd78dbd9841dc355f1afc510))
- forward Open WebUI user-identity headers to BASE_URL ([#50](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/issues/50)) ([`a6918d7`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/a6918d74e3f9f96d32c0d4c220ec48a1082bc9ca))
- **streaming** — adopt Open WebUI 0.10.x native reasoning output items ([`623e21d`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/623e21d32c1b2f72ab07d35d178a122a1c641320))
- **valves** — default tool-result persistence to off ([`bb5766c`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/bb5766ca2a9dbb96974d8dd860d36dac1c7dcb33))
- **dashboard** — add self-update tab with snapshots, variant choice, and auto-update ([`9ea3df8`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/9ea3df8883f672dfd7e49073e3aaa4531dff1422))
- **fusion** — stream panel answers and reasoning live with in-card thinking view ([`ae1bae7`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/ae1bae783f01376db42adb965223f41e72940bdf))
- **fusion** — add built-in deliberation engine with switchable backend ([`253423b`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/253423bb564127f253ec2798f39ae5e9dd3bbda3))
- **images** — wire the per-model help command for image models ([`f61ade4`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/f61ade45da1dbd5bd46c1b5d763fb1b2da2858ef))

### 🐛 Bug Fixes

- **runtime** — gate anyio #1111 workaround on anyio < 4.14.2 ([`1eabbdd`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/1eabbdde8175f58dc45a503073bc04089ae802a5))
- **models** — recognize ~ router-alias model ids in provider matchers ([#53](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/issues/53)) ([`fd161de`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/fd161de4198449d345492a06446a29a4ffb3430e))
- **streaming** — preserve canonical tool call IDs ([`77c35d1`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/77c35d1c944d7ba4a793bdb772361a01fca622a3))
- **dashboard** — harden update writes, valve gates, and redis lock cleanup ([`419aeaf`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/419aeaf9c4f268c6fc92e2e2fb227f3c39750471))
- **dashboard** — correct embed-position hint in the dashboard message ([`7ee1561`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/7ee1561e5d18c28dc3d6f253e8a56d576c648c67))
- **fusion** — guarantee deliberation on fusion models ([`d2e6cf5`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/d2e6cf5a23d62391a20e1c906c0164f722a9143c))
- **tests** — satisfy pyright attribute checks in fusion engine tests ([`32d5fe1`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/32d5fe1580d53a025f5dd7adde5ad543bcdf4e94))
- **docs** — revert fusion doc changes ([`b279a31`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/b279a31f40616e529478f1db1c329801b0612ea4))
- **changelog** — skip version-bump commits in git-cliff config ([`91026bb`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/91026bbae2e3055f97eadc4644c6409714166c77))

### 📚 Documentation

- document plugin system and pipe_dashboard; refresh valve atlas ([`ef15995`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/ef159959080945b2dc6764e57f314fd7a807e16a))
- **fusion** — document guaranteed deliberation and live panel streaming ([`230d434`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/230d4346dbb25cc3f0f4252d61e99fffe338a5e3))
- add fusion screenshot ([`d070892`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/d070892e1f9006c3e3617df54f975af6a0f3b5e1))
- regenerate changelog and fix git-cliff config ([`fbf95e2`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/fbf95e27b56b2fdaf8b4606e2158738102b6553e))

### 🧪 Testing

- **pipe-dashboard** — fix dead-pubsub publisher test hang on Python 3.11 ([`ad9dd12`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/ad9dd125c741460396edcaf1687306b19718457b))
- **reasoning** — fix pyright errors in vendored OWUI replay harness ([`1e8a629`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/1e8a62974acf50bcba63a86142c10cd7c964233b))
## [2.6.9](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/releases/tag/v2.6.9) — 2026-06-30

### 🚀 Features

- **caching** — stable per-conversation session_id to maximize prompt-cache hits ([`516697a`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/516697a236bb0b2bb610aa93fb4c3b9808b753fa))

### 🐛 Bug Fixes

- **runtime** — extend anyio #1111 workaround to cover 4.14.1 ([`a5c989d`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/a5c989d120b6ad3dd6179f2e5f4d4f00dd0d28d1))

### ⚙️ Miscellaneous Tasks

- ignore .env ([`5043d00`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/5043d006d0c922f5df041e0de03aee17bcf0ddcc))
## [2.6.8](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/releases/tag/v2.6.8) — 2026-06-29

### 🐛 Bug Fixes

- **reasoning** — resolve Anthropic thinking-signature 400 on multi-turn tool replay ([`fde0ea7`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/fde0ea7b6eb968323965dadb9a38342bdb515929))
- **reasoning** — treat blank/non-string thinking signatures as absent in Anthropic strip ([`4df0581`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/4df05817083143203bd940155988c18738de0afe))
- **sanitizer** — log proactive Anthropic reasoning strip (was silently unlogged) ([`60dac53`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/60dac53aa133f78ebc0ecc13a2f2b9d6a9de4e0c))
- **fusion** — heal live judge-analysis spinner and surface item sources as citations ([`cc5ab78`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/cc5ab78cee91cae4d18e1884b489595ad738afc1))

### 🚜 Refactor

- **cache** — remove dead, blind Anthropic prompt-cache retry fallback ([`39de58c`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/39de58c02b4c5d01b1654fdc4044187a2d4d56c7))

### 🧪 Testing

- **reasoning** — cover fragmented reasoning.text consolidate->replay round-trip on chat ([`34d7318`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/34d7318e9631a45153eadf40f8cd35edeee027a4))
## [2.6.7](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/releases/tag/v2.6.7) — 2026-06-27

### 🚀 Features

- add OpenRouter Fusion with live deliberation panel ([`945fc7e`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/945fc7eda93e460885a6636e7ae69e05d79c1b3d))
- **models** — Gemini Flash 3.x image filter + HappyHorse video help ([`f16493e`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/f16493e37ea43a0ac343d290584847f3271e4c60))
- **server-tools** — add advisor, subagent, and model-search server tools with cost guard ([`de5f532`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/de5f53251684da7ec9fc5896427befc81026faae))
- **citations** — warn on unrenderable citation types (e.g. file_citation) without breaking the answer ([`0a3fdd5`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/0a3fdd5eaf11e7e7a2561715d2dd100bf2f2cead))

### 🐛 Bug Fixes

- **reasoning** — preserve thinking-block order across tool rounds on replay ([`99788bc`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/99788bc927a4c1344c38405389f600c9391315d7))
- **storage** — backend-agnostic, authorized OWUI file gateway ([#46](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/issues/46)) ([`1471288`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/1471288fb16a59fc673e137c06ce89c6047f4783))
- **timing** — drop @timed from Pipe teardown chain (finalizer log pollution) ([`28356de`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/28356de5e28096082330be4f17d5980859bd2507))
- **server-tools** — render openrouter:* items end-to-end and harden the web tools filter ([`b1db107`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/b1db107742e1deebf41dfaa123aaa964e06bb93e))
- **citations** — correct url_citation excerpt, superset-drop, and web_search/web_fetch source extraction ([`256a7e6`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/256a7e6a847099d289592bf8390f6300845da061))
- **responses** — Claude prompt caching + interleaved-thinking header for Anthropic on /responses ([#48](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/issues/48)) ([`d448d45`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/d448d4540d98f2d04532bf1b0c83b2676d01b6aa))
- **context-compression** — emit plugins shape instead of deprecated transforms:[middle-out] ([`b7454f2`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/b7454f2eb1eda2d011312e884b06c08e3da1c3e8))

### 📚 Documentation

- feature OpenRouter Fusion in README + refresh model/test counts ([`4757a35`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/4757a352923e746faa492179c7465c9242531cb6))
## [2.6.6](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/releases/tag/v2.6.6) — 2026-06-15

### 🚀 Features

- **image** — support MAI-Image-2.5 and Riverflow 2.5 models ([`9f1d59e`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/9f1d59ef4f8d3bb19e4710d117341ef53e125f3a))

### 🐛 Bug Fixes

- **filters** — gate image filters on prefixed model ids ([`c3a07dc`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/c3a07dce5016f1e816c883d881dd10e1c4ac7550))
- **errors** — guard non-dict error metadata ([`3139f31`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/3139f313a94a97697087e5066d18b57deb7ca36b))
- **tools** — guard non-dict tool arguments ([`0f72651`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/0f7265162582d2c707175b6ffae93f6033751381))
- **reasoning** — disable Gemini thinking at budget 0 ([`68f2ae5`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/68f2ae5779303d60c7af50a1b3a3e5c32f1d63ca))
- **streaming** — correct www. strip in citations ([`42ac292`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/42ac292e46e1cbd8906498c7493205af863d5e4f))
- **requests** — preserve native audio formats ([`c40bfcf`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/c40bfcf140a52ab12a0482fa230efb57b529c2e6))
- **api** — stop stream retry after first emitted event ([`e619279`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/e619279a3775634d43f61423fe0239ac82f02a90))
- **storage** — keep cached artifacts when breaker open ([`491463e`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/491463e952ae9b23ce23e3bb54ac5f458f74ef74))
- **api** — parse HTTP-date Retry-After ([`3444bd2`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/3444bd24239523a4b2354ee45ebcac5caf07717b))
- **video** — read frame_type on retarget ([`750f565`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/750f565d7b5a464a8ba4cbc66433a485cec1651c))
- **video** — clean up per-job temp dir ([`c759264`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/c759264809c053b1f0a1396f5133a0f74e2fb6e3))
- **logging** — cap timing-event buffer ([`36c43a4`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/36c43a4b5f869b121c4a1cda853cd866ecc68405))
- **video** — offload passthrough URL validation off event loop ([`3bfc9fe`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/3bfc9fe4991de33e149a6e56b1d0c96c17c5075f))
- **media** — clamp ffmpeg seeks past EOF ([`2e37c71`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/2e37c71ef35bc8158406da890127740248336786))
- **video** — match indexed downgrade codes ([`59c433a`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/59c433abbb564bf0f0bde3701e4a17a8309d3264))
- **logging** — always write jsonl so text-mode archives merge ([`0e953ba`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/0e953ba9e72f7e49f3896e2777aecbd714c3a233))
- **logging** — dedupe archived events at ms precision ([`e60e6f1`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/e60e6f1a24b00f0a15eb9ea9c945ef7fad8bf42b))
- **security** — pin remote downloads to validated IP ([`b35f6e2`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/b35f6e261f9540628f43ca4269243da0a6560477))
- **pipe** — parse HTTP-date Retry-After in 4xx handler ([`0f6bac7`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/0f6bac7690ef5ab06a0bcfedca30a230d2fbbf46))

### 🧪 Testing

- **logging** — assert worker messages survive concurrent writes, tolerant of stray lines ([`176879c`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/176879c8e72115ebdd1580b4f836466725b5c21f))
## [2.6.5](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/releases/tag/v2.6.5) — 2026-06-09

### 🐛 Bug Fixes

- **catalog** — update OpenRouter frontend models URL to new catalog endpoint ([`9a298fd`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/9a298fde1ba76dba20335479b6e01ed5898c4086))
## [2.6.4](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/releases/tag/v2.6.4) — 2026-05-19

### 🚀 Features

- **image+video** — support xAI Grok Imagine image and video models ([`22340d9`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/22340d99d46ec577229031c1413b43bc10d724cb))
## [2.6.3](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/releases/tag/v2.6.3) — 2026-05-18

### 🚀 Features

- **image** — support Recraft v4.1, vector, utility (8 new models) ([`ef9d472`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/ef9d472328d88a9a7deee95aa1e74e3af245f2bf))

### 🐛 Bug Fixes

- **timing_logger** — use RLock so @timed re-entry doesn't self-deadlock ([`33d9b5b`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/33d9b5b093d8be4ffa297b1c15d85c5c52ce0308))
- **logging** — recover task model message_id on OWUI 0.9.x ([`30cb3ce`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/30cb3ce32ff5e3e4489e1bfb9add54fbfb8b3f30))
## [2.6.2](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/releases/tag/v2.6.2) — 2026-05-17

### 🐛 Bug Fixes

- **bundle** — anyio #1111 workaround no longer mutates _tasks ([`bb841ed`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/bb841ed69f2a92cfd66d2cdeff858258a1b7fb43))
## [2.6.1](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/releases/tag/v2.6.1) — 2026-05-15

### 🐛 Bug Fixes

- **bundle** — backport anyio #1111 workaround into both bundles ([`2d5e36d`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/2d5e36d6cea998dd9f278619d5117e5221063d81))

### ⚙️ Miscellaneous Tasks

- revert 3.12 matrix and verbose pytest to locate hang ([`baae29f`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/baae29fdab41c637c7d5fb25445c53d0f71e61ff))
## [2.6.0](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/releases/tag/v2.6.0) — 2026-05-12

### 🚀 Features

- **video** — intent classifier for stateful video requests ([`f87d1ee`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/f87d1ee1a02dbe0b1a2f74552725fbaeb85fac7d))

### ⚙️ Miscellaneous Tasks

- harden bundle loader + Python 3.12 CI matrix ([`2da767f`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/2da767f890d84758c51ae1b025ea3768d1faf56a))
## [2.5.3](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/releases/tag/v2.5.3) — 2026-05-09

### 🚀 Features

- hot-reload pipe lifecycle without OWUI restart ([`7c667ce`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/7c667ce6c1552f78d7b1605b5e6e75f703692793))
## [2.5.2](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/releases/tag/v2.5.2) — 2026-05-08

### 🚀 Features

- **image** — support Recraft V3/V4/V4-Pro with 2 new filters ([`aaa7ec5`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/aaa7ec55d95001ed3e2dbf4ae88ac463362dd907))

### 🐛 Bug Fixes

- **video** — always re-raise CancelledError in _safe_emit (anyio #1111 trigger) ([`9520f2d`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/9520f2d4c787bc2d459b24d8219c9206f71cce2a))
## [2.5.1](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/releases/tag/v2.5.1) — 2026-05-05

### 🐛 Bug Fixes

- **image** — video parity - preserve on chat refresh, ensure from pipe() ([`76eb78c`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/76eb78cbea737f1494993c70667f42516b284729))
## [2.5.0](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/releases/tag/v2.5.0) — 2026-05-04

### 🚀 Features

- **video** — support Kling v3.0 Pro/Std with cfg_scale knob ([`a8ea0e1`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/a8ea0e14be4437984a8ba680efed42828e73466c))
- **image** — native image generation for pure-image-output models ([`9a1a0ae`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/9a1a0ae701ef5b99ec12ecc61b00e6e1b23855d7))

### 🐛 Bug Fixes

- video catalog TTL + ZDR fail-closed (security) + atomic register ([`1f75dc1`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/1f75dc199673c508d6fd7240ed4d77fa3db81739))

### 📚 Documentation

- clean up broken local references ([`e16d82f`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/e16d82f35fbfcf771eff99ffff72d240d3acd21a))
- image generation feature doc + README refresh ([`ba1f696`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/ba1f696de5bd67da9c68fbc612258bed8875dc10))
## [2.4.2](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/releases/tag/v2.4.2) — 2026-04-28

### 🐛 Bug Fixes

- function_call.name must match ^[a-zA-Z0-9_-]+$ for OpenAI Responses API ([`dde7053`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/dde70538523870c8daecb3e8f3841223990fea19))
## [2.4.1](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/releases/tag/v2.4.1) — 2026-04-28

### 🚀 Features

- replace OpenRouter Search plugin with server tools (Web Tools + Image Gen filters) ([`c446ff9`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/c446ff94bfb806826f6ec53661fe1ed7315a3017))
- add tool execution cards for OpenRouter server tools (datetime, web search, web fetch) ([`c88c230`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/c88c2308ef55ccf3ebd7ef64661ecf70d8550930))
- add OpenRouter async video generation (11 models) ([`d868390`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/d86839026be445ecc589294180914d5c056321e8))

### 🐛 Bug Fixes

- server tools post-merge fixes (merge pattern, image gen UX, streaming handler) ([`eb4c73b`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/eb4c73b1062c7e0adbe46a30b9648666912bff31))
- remove CSS patch — race condition in OWUI 0.9.x causes chat freezes ([`11290c6`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/11290c67490a5a1c8a4b8e5f91bf50ae160f08c2))
- shorten filter display names (OR prefix) and default web tools to off ([`a3aaf29`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/a3aaf29b177a0286bc81c94098e8c8529f469d68))
- add thinking_config to API whitelists and fix variable shadow in top_k handler ([`ae59dc2`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/ae59dc20c26702350997a7e067816509d53d0c06))
- complete image gen spec and harden image output handling ([`8a0cc0c`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/8a0cc0c3520c632b1b511cead05f25ed4a2f0fd6))
- web search tool card shows citation pointer (result data not in tool item per spec) ([`2a0f3bd`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/2a0f3bd81839c4834b8b1c76dc72ca4c5a359d7d))
- video parity — descriptions, web_search overlay, usage emit + Redis billing ([`61aab23`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/61aab23c45e89a6e83b40e451994c9091a2be633))

### 🚜 Refactor

- centralize filter metadata namespace key into _PIPE_METADATA_KEY constant ([`4c3964d`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/4c3964d2a8500c051cc0d92d272515ded4515147))
- unify tool card emission, gate server tools by SHOW_TOOL_CARDS, fix bugs ([`2c5cba6`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/2c5cba63e9979e95f1083955ae9ab0a1d75da6de))
## [2.4.0](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/releases/tag/v2.4.0) — 2026-04-23

### 🐛 Bug Fixes

- inject OpenRouter web plugin for all models, not just those with native web_search pricing ([#26](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/issues/26)) ([`c0c3f8e`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/c0c3f8e7a655c55b581e29f36792220f6d6608a7))
- migrate all OWUI calls from sync to async for Open WebUI 0.9.x compatibility ([#32](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/issues/32)) ([`b764d41`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/b764d41eb798996204a3db38f95414315c987e9e))
- populate filename on file blocks sent to OpenRouter ([#31](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/issues/31)) ([`ee127cf`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/ee127cf868bd3cb47fd709a4a550c31b5cb1f1f2))
## [2.3.3](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/releases/tag/v2.3.3) — 2026-04-06

### 🚀 Features

- per-model provider routing via OWUI Advanced Parameters ([`7b32247`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/7b322470eaddafc78e243ba9ccb37e6df41094d1))

### 🐛 Bug Fixes

- add explicit permissions to CI workflow ([`7b45793`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/7b45793a7db5d2d2cebce3b3b58a47b499984f20))
## [2.3.2](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/releases/tag/v2.3.2) — 2026-04-04

### 🐛 Bug Fixes

- support non-catalog model id fallback ([`0b43820`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/0b4382045bf0da2c0347ca29d97acf8b3397c3f1))
- prevent resource leaks on request cancellation ([`e5fd4bc`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/e5fd4bc4da44ef11dfa96cb5cf24598c6d974cf2))
## [2.3.1](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/releases/tag/v2.3.1) — 2026-03-31

### 🐛 Bug Fixes

- respect AUTO_CONTEXT_TRIMMING on Responses API path ([`1cae605`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/1cae605c3c35b5e6e8e652e53b3ebdb5cb63000e))
- validate function_call/function_call_output pairing before API send ([`7c26df2`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/7c26df2159d742aa49fe542f10ff21c9b632c882))
- route MOA tasks through normal chat path ([`a770ac2`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/a770ac2382ae4d7550a4960b911e160eb4746b72))

### 🚜 Refactor

- reuse OWUI artifact payload sanitization ([`a3e4455`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/a3e44557ab5a86daa1757cdb60d4ddf044794aa2))
## [2.3.0](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/releases/tag/v2.3.0) — 2026-03-14

### 🚀 Features

- add Nagle-style adaptive delta coalescing for streaming ([`a903e47`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/a903e477fb4b9e27eb57a547ab9d445b10847033))
- preserve GPT-5.4 phase across OWUI history ([`9caf3d4`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/9caf3d4838160353998d401c9319f006a258579b))

### 🐛 Bug Fixes

- handle lone Unicode surrogates in artifact persistence ([`47e8097`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/47e809777645d8bf59742110aef21d8a4e912666))
- preserve Responses item order in pipeline continuations ([`4308aef`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/4308aef787885e4d4b1ba449fa78f6fe339d5d5b))
## [2.2.11](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/releases/tag/v2.2.11) — 2026-03-05

### 🚀 Features

- two-phase tool card lifecycle, incremental updates, and passthrough fixes ([`e2551f5`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/e2551f509654a6dde3c1a05b66f8d6d05e0334e9))
## [2.2.10](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/releases/tag/v2.2.10) — 2026-03-01

### 🚀 Features

- add OpenRouter trace/Broadcast observability support ([`af3ae25`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/af3ae255ea59d504ef2e86c59a325ab6e4761ff5))
- sync Responses API allowlist with live OpenRouter OpenAPI spec ([`a23723f`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/a23723f85a273885e4cf0e293ccdadd430694c3b))
- preserve cache_control on tool definitions and add adaptive Anthropic breakpoint injection ([`7162834`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/71628341649b28f23443ce9444106733643635ff))
- map xhigh reasoning effort to verbosity: "max" for Claude Opus/Sonnet ([`a555e6c`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/a555e6c5e80040af972dce702b0e83de4d211fff))

### 🐛 Bug Fixes

- migrate X-Title to X-OpenRouter-Title and add X-OpenRouter-Categories header ([`5c8e2a4`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/5c8e2a41502f72373965a7310e8abddb4080ff8d))
- unify HTTP error handling with template-based error messages ([`29bb947`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/29bb9477ea4e23dfc58b3f87a94d27b5153f387b))
- remove context saturation guard that incorrectly stripped tools ([`40ae3ce`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/40ae3ce184b2ecb5dfe3c1f52de4e16d53934451))
- graceful tool-loop limit — inject stubs instead of hard-cutting ([`51a0bcb`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/51a0bcb0c59df3053b6111d0376ce94442920e65))
## [2.2.9](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/releases/tag/v2.2.9) — 2026-02-28

### 🐛 Bug Fixes

- properly close unawaited Pipe.close() coroutine in __del__ ([`dd1cafc`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/dd1cafc0fbb63b75121eec901f7a90f6b7d36108))
- remove arbitrary 50% fraction cap from tool output budget guard ([`3c6b4b1`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/3c6b4b15590ceb92618077372e166d07fb428c41))
## [2.2.8](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/releases/tag/v2.2.8) — 2026-02-28

### 🚀 Features

- add adaptive context budget guards for tool loop resilience ([`6c7b613`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/6c7b61335e9cc4c01897bcd95c4810fad319c0eb))

### 🐛 Bug Fixes

- support OWUI access_control schema in model overlay sync ([`b5e06f9`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/b5e06f964d62b7302f37b62b033da2fa4b23a102))
- close reasoning timing before tool card emission in pipeline mode ([`f2bcc32`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/f2bcc329831107b14c7eb86b0db878b8d7548e7b))
## [2.2.7](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/releases/tag/v2.2.7) — 2026-02-23

### 🐛 Bug Fixes

- enforce strict variant IDs for model restriction checks ([#16](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/issues/16)) ([`4e61467`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/4e61467a9b87c083cfe0a29a714f8738f8c95671))
- line-anchor tool card <details> blocks for OWUI markdown rendering ([`f98ae45`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/f98ae45e9610660d7bd66448e19ea1c681058a54))
- avoid duplicate tool content in source tags ([`9c1f641`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/9c1f641385428f855aacfcfc59934ef5b68005e2))
- align tool streaming with OWUI response events ([`48cb036`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/48cb036af5308a50b9110631c18ad506e4a87016))
## [2.2.6](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/releases/tag/v2.2.6) — 2026-02-18

### 🐛 Bug Fixes

- preserve tool context and handle malformed calls ([`5a4ea96`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/5a4ea9649e122b1a2dff24e35c76a013379724fa))

### 🚜 Refactor

- remove defensive valve access patterns ([`029436c`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/029436cfb07b686a48908e09c0c475d6798bb3c3))

### ⚡ Performance

- enable SSE delta batching to boost streaming performance improvement ([`29ab8f5`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/29ab8f5f6475ef85c2f13aaf578d88322b45189f))
## [2.2.5](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/releases/tag/v2.2.5) — 2026-02-18

### 🐛 Bug Fixes

- protect all unprotected event_emitter calls ([`af7fdd3`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/af7fdd32b7f1a80377d9712a79c9c47aa2bfbf8b))
- prevent distributor hang on malformed JSON chunks ([`3bb715b`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/3bb715b4f5e7a86cae2a7ff67bf01e03817aa834))
- wrap entire pre-enqueue section in safety net so no exception crashes the pipe ([`fb86844`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/fb86844e80d36d17708a70daa1e2b17c234b6907))
- replace hardcoded SessionLogger.max_lines with valve-driven default ([`91aec23`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/91aec23e43b727d08eccb698010149ed06fa6b03))
- size DB thread pool from engine connection pool instead of hardcoding 5 ([`471c0e7`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/471c0e7aefd3f286966820cc3a5c30178f598a78))
- prevent distributor hang on malformed SSE chunks in responses_adapter ([`7ed15ca`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/7ed15ca521689f932c45676c86cab7f016e97a45))
- avoid sentinel deadlock in responses_adapter producer cleanup ([`fd5ee28`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/fd5ee2810b5dc96f2f424c693c4c7666aff362e8))
- add chunk_queue backlog monitoring in responses_adapter ([`963f9af`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/963f9afea6b136a44a1356048576217640197b32))
- harden file inlining and SSE parsing in chat completions adapter ([`d897679`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/d897679e6e93ce3ff44baecac10e400de8c02ac3))
- drop middleware stream items on timeout instead of killing the session ([`0cad2a1`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/0cad2a195dbb72a6f0bcef261fd5a1994e695c5c))
- attach exception callback to catalog metadata sync background task ([`a9687ce`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/a9687ce914e606a6810f18e1d98fcd40548ae4a4))
- allow requests through while warmup recovery is in progress ([`8158c60`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/8158c60dcc1780513dbb30da93672cb2f9d53144))
- preserve partial response on interrupted stream ([`46df490`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/46df490051a771f83ace1f5c36030a3508b954c9))

### 🚜 Refactor

- inline _format_openrouter_error_markdown into its single caller ([`56b9fc5`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/56b9fc54e72f7214c1090b8b1d8a2d98a483780b))
- remove dead SSEParser class and its 33 tests ([`34b30a3`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/34b30a36df1661287b96404dce29d6ab9bc1c4ba))
## [2.2.4](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/releases/tag/v2.2.4) — 2026-02-16

### 🐛 Bug Fixes

- prune stale openrouter_* filter IDs to prevent OWUI 0.8.0+ startup crash ([`b886a92`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/b886a92b242ef34092944b83e973674ef465cf5a))
- return tool failures as outputs instead of raising exceptions ([`6fb976a`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/6fb976ac0a0577c19cc122085777e628606935f8))

### 🚜 Refactor

- remove dead ReasoningTracker class and its tests ([`5c6b043`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/5c6b043ff55539b051ddb3025b3d07dfde1c31fb))
- extract ReasoningStatusThrottle to deduplicate reasoning status logic ([`9d7429b`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/9d7429be938844d833bc0cd1d95f873aef008fa4))
- deduplicate _inline_internal_chat_files in ChatCompletionsAdapter ([`6e5a021`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/6e5a021cffe0fb2292a03b6807c39265e8f55da9))
- extract _ensure_filter_installed to deduplicate ORS and Direct Uploads filter lifecycle ([`a9e922a`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/a9e922afa5d48084fae6bdaf8fb5c229ba1834be))
- extract _db_session context manager for consistent rollback and close ([`134c7eb`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/134c7ebc813df70eae7c22fef98f8e4419f05c28))
- extract _detect_redis_config to deduplicate Redis init logic ([`93104c0`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/93104c00965557def08cb3050883709bc9a29dbf))
- extract _parse_url_citation_annotations to deduplicate annotation parsing ([`755338f`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/755338f63933085c67d52193c49ab48ad005c42a))
## [2.2.3](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/releases/tag/v2.2.3) — 2026-02-14

### 🚀 Features

- add bundle_v2 flat monolith bundler, lint and type-check bundles in CI ([`c916ae7`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/c916ae7b2feaee467b08a52ab586db90b34147b9))

### 🐛 Bug Fixes

- complete mock_valves fixture with missing valve attributes ([`20acfb1`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/20acfb18892fa523ea06be84cd6d92e404762d5a))
- rename shadowed loop variable in reasoning_tracker (ruff F402) ([`2795dd7`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/2795dd7b5cd87759fc1255451c6b8f30c0536c65))
- mock _download_remote_url in image tests to prevent real HTTP calls ([`4ab6253`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/4ab62536906e2728bb9217e099b0b5bdeeeacd2f))
- deduplicate _RedisClient type alias — single source in persistence.py ([`7edf912`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/7edf912d7d45ee36432dc461def3c947bfd4ff22))
- harden bundle_v2 — exit on validation failure, tokenize-safe alias replacement, robust else: detection ([`dbf5fb1`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/dbf5fb16271261c4fceb21f7bd2d6eed549a8251))
- graceful DB error handling in session log assembler ([`898eb19`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/898eb192395129745cf88a3aa7614642dda07a66))
- skip non-replayable artifacts in persistence gate ([`0df7398`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/0df73986dd5b391b06c36089de6aa3207d5ebf7f))

### 🚜 Refactor

- consolidate normalize_persisted_item into single function with default parameter ([`dac27d5`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/dac27d5d5bde573ecdd0a727376cb82b3599afa5))

### ⚙️ Miscellaneous Tasks

- finalize modularization — deduplicate names, clean imports, add ruff to CI ([`556afa2`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/556afa2f770b66429e5f75b51bb5cd7b440a8eac))
## [2.2.2](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/releases/tag/v2.2.2) — 2026-02-13

### 🚀 Features

- OWUI 0.8.0 compatibility — migrate access_control to access_grants ([`50094b9`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/50094b9487df2e4a4ad6e7ed6a86f9689e4d3d49))
## [2.2.1](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/releases/tag/v2.2.1) — 2026-02-09

### 🐛 Bug Fixes

- prevent token duplication and restore citation parity with OWUI ([`9c096bb`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/9c096bb78e5e0f34a69f54411e5d2c1507e094f6))
- handle end-only citation annotations in streaming ([`f710bb3`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/f710bb307902c4741d9cc49616580a0421900fe7))

### ⚙️ Miscellaneous Tasks

- update release checklist and manifest pin ([`07407a4`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/07407a4b460942bf6fbac3a5feeaae6f0235a4e9))
## [2.2.0](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/releases/tag/v2.2.0) — 2026-02-03

### 🐛 Bug Fixes

- relax typing for usage icon set ([`f532db1`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/f532db1184790a28c28989c9d22d60ec5e3b3482))
- pass reasoning encrypted_content back on tool call continuations ([`db3e781`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/db3e781be475b392e837b9df089de2934e10bb25))
- pass reasoning encrypted_content back on tool call continuations ([`c1d53a5`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/c1d53a59b30529e932243fb10fe721e1def1079f))

### 🚜 Refactor

- modularize pipe.py into domain packages ([`9e9dd50`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/9e9dd50328b64f89fec91ab4892db52914b8a528))
- complete modularization of pipe.py into domain packages ([`bd43385`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/bd43385272a2afd3d7d0aa0ce89de289c7b00ec1))

### 📚 Documentation

- add ZDR links to README and docs index ([`f8027e5`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/f8027e5c1f79f257e4d498442b0b4c561c02e7b5))
- update bundle install references ([`bc82dfb`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/bc82dfb171133f4e523df1a5f349d8563f8efbd5))
## [2.1.0](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/releases/tag/v2.1.0) — 2026-01-31

### 🚀 Features

- add icon/text styles for final usage status ([`a7c986a`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/a7c986a025f29701d75445ab7f0737a6bf838e93))
- add OpenRouter ZDR filtering, enforcement, and catalog flags ([`2bb32db`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/2bb32dbe9b0eb7380ed13564b4a63ce3449414b5))
## [2.0.7](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/releases/tag/v2.0.7) — 2026-01-31

### 🚀 Features

- extend OpenRouter usage mapping for new detail fields ([`fccc2f5`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/fccc2f5f1ad1a8f96eeff29ab3972fb5d04dcec2))
- extend OpenRouter provider-routing max_price valves ([`8b35be9`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/8b35be91b2540336b2bd5c07eb2aef62355c906a))
- add OpenRouter PDF parser valve for direct uploads ([`ebfd258`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/ebfd2582fdd4d21c4849f977194d11aaaae0f266))
- map OpenRouter reasoning fields to OWUI thinking ([`f732b49`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/f732b497258a3c10dcd7a0a71018dbef25ec2a05))
- map OpenRouter chat images into output items ([`9b4615c`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/9b4615cb3ce48f42ee9af150ea847d729a2b3084))
- surface chat refusal text when content is empty ([`393cbf2`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/393cbf2d825320bf8e6194db8ca496869d5eec09))

### 🐛 Bug Fixes

- OpenRouter provider routing catalog variant handling ([`e4cade5`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/e4cade558b8a1fb321efd745cf8f494940e4e38e))
- harden pricing parsing for new OpenRouter schema ([`878c44f`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/878c44f091c7c47840ced054dfe2afc464176d51))

### 📚 Documentation

- note response-healing is not exposed ([`23b17ba`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/23b17ba6575be4db29ecd4c922c14e0c251a34fa))
- note auto-router configuration lives in OpenRouter UI ([`d46037c`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/d46037c1452c10e4dd5404e6a1a7f619ccf5bd0e))

### 🧪 Testing

- cover OpenRouter file annotations ([`26429be`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/26429be8b9332b21998b9187e83885bec1b14cb7))
- verify developer role and usage details passthrough ([`ad8fb6f`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/ad8fb6f278d7ea1fcac4dad40ba72ec7eb81d047))

### ⚙️ Miscellaneous Tasks

- ship dual bundles and test them in CI ([`cd672a7`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/cd672a7c0d9533279def552dd2ad83fe1dccd726))
- align OpenRouter usage flags with docs ([`f9d4bf9`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/f9d4bf92148c0701b1c1518b9bd37b734ad65a41))
- OpenRouter Gemini-3 reasoning cleanup ([`a74c43d`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/a74c43d43b571bc11d20836ebe6b495e7a9a4d0c))
## [2.0.6](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/releases/tag/v2.0.6) — 2026-01-28

### 🚀 Features

- tool backend parity, SHOW_TOOL_CARDS valve, fix image generation ([`cce0565`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/cce05659c8b2ad9d77915ebd9ec16b6cfcf74b1b))

### 🐛 Bug Fixes

- use getattr for rowcount to satisfy pyright ([`5ab88e6`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/5ab88e67bce5939c8da7d6062f2eee04734ca649))
- add fetch-depth: 0 for release notes generation ([`9e2a97a`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/9e2a97ac48fb954188988e39dde16db2b64dcf2a))
- tool cards use correct event types for ephemeral/persistent states ([`d845011`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/d845011f8f9495b4a3bc415e6086b08d2e96f1dd))

### ⚙️ Miscellaneous Tasks

- rename job to 'Tests & Type Check' ([`8f1ad78`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/8f1ad7837dfa0fd491b83f57e2d7a49300afe17d))
## [2.0.5](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/releases/tag/v2.0.5) — 2026-01-26

### 🐛 Bug Fixes

- make new models private by default (fail-safe) ([`3e3a0ff`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/3e3a0ff16a607c96930372f8689bb8bcdae490c4))
- add type guard for optional error in test assertion ([`b2d9b53`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/b2d9b53c0341d78d4c7e9d4e65763e840a5e2470))
- use upsert for session log lock to avoid duplicate key errors ([`a53f666`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/a53f666dd84b42945a6c158ec2e2ae4f33726491))

### ⚙️ Miscellaneous Tasks

- use uv for faster dependency installation ([`98504ec`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/98504ecc2a290f0a015c0146545320155aa4f971))
## [2.0.4](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/releases/tag/v2.0.4) — 2026-01-25

### 🚀 Features

- add dynamic per-model provider routing filters ([`c1ede06`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/c1ede06f508417663c9929c1aab41acb88973a18))

### 🐛 Bug Fixes

- add 404 to special_statuses for OpenRouterAPIError formatting ([`3a2c8e3`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/3a2c8e3b0c7115638f1d03b7448f988aee74befd))

### 📚 Documentation

- update install URLs to use bundled releases ([`a6904ad`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/a6904ad8804c08adbecd6fbad231e2d63a55fee6))

### ⚙️ Miscellaneous Tasks

- generate release notes with clickable commit links ([`4f406d2`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/4f406d2f8505fdce251992b3b3db6f54eb0ea563))
## [2.0.3](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/releases/tag/v2.0.3) — 2026-01-24

### 🚀 Features

- Implement major overhaul for concurrency and resilience ([`2d8e450`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/2d8e450f01d9091b422098a87de15c45ed01bee6))
- inline hosted images and tighten capability plumbing ([`4de1fa8`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/4de1fa8659b9ff049741df8308e6b661f67d98f4))
- add gemini thinking config and reasoning docs ([`f8313b0`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/f8313b0852660bf15348592d85ed47d3aa339a41))
- encrypt redis artifacts ([`00a2fdb`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/00a2fdb8d0c4478dd0c893b24c13dad6e3968779))
- add stream emitter and thinking output valve ([`1ca4920`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/1ca4920f9bd17a52c6776aa4b2a967ebe9a96d89))
- add valve-gated request identifiers (user/session/metadata) ([`f7e2d2e`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/f7e2d2ef3fb8996833757efe0b29d74f0d245cdb))
- add encrypted session log archives for abuse investigations ([`362f29c`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/362f29ca5d99de7adae1a6fe0e4396160435e4fc))
- support model fallbacks via models[] ([`b9c4e59`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/b9c4e5956800453f9382beea26972e20ba2ece49))
- pass through top_k when provided ([`1befb4b`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/1befb4bff1e6514fd334dbe7d281cd02a1efdaac))
- support OWUI Direct Tool Servers (execute:tool) ([`5db0db5`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/5db0db564078ec56e1cdcdcbcf024583e02a537a))
- add HTTP_REFERER_OVERRIDE for OpenRouter headers ([`8503c78`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/8503c78b203ff4228342ba5d806ca6f99a29912e))
- warn when tool loop limit reached ([`27b89b4`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/27b89b4c7ca15a6650004d3894989a593e74da0c))
- enable Claude prompt caching ([`aebb2df`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/aebb2df89d7ccd490325d02260cbb1739b1211d6))
- sync OpenRouter model metadata into OWUI ([`270c587`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/270c5879690c41081346515039794fef8ae9299a))
- add disable_native_websearch custom param ([`523b431`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/523b431da141ac9759bb8e667d4ae2dbe3a88a9f))
- preserve chat-completions params for endpoint fallback ([`221e56a`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/221e56a64ba3d625d0f0c9a68a2a82e3bbf3a288))
- extend artifact retention on DB access ([`fa218f6`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/fa218f609edabe664a5b4c6983ae82e35d39cb20))
- add jsonl session log archives ([`ce329d4`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/ce329d47ddd725c7d6ef2f034cbf85fe1b3718e0))
- major web search rework + OpenRouter Search defaults ([`f73ea2a`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/f73ea2a7c6939b2bc8ba5249123bfc184c5f6555))
- add Open WebUI stub loader pipe ([`c2d8bfe`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/c2d8bfefe6b1e8300953dca93bca169a6cd69c26))
- support Open WebUI v0.6.42+ chat_file tracking ([`6f8bff7`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/6f8bff7094ded9d708e98fa094ec337073c67ebf))
- add model filters for free pricing and tool calling ([`aa7df3e`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/aa7df3efd8b335b43ab5950bbdc1b01d934933eb))
- add Open WebUI tool execution backend ([`bf7c5ae`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/bf7c5aeb5bf97509b38b73c60e3b7e0a50e97f23))
- add Direct Uploads (multimodal chat uploads) ([`5b832e6`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/5b832e6b7cb5ec318a25246bf3034b5db05bdd5f))
- support OWUI 0.7.x native tools + builtin tools ([`e27b912`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/e27b9125d203ebaaea9ecd93f11949e7c8a9aeae))
- auto-discover OWUI DB engine (OWUI 0.7.x DB refactor) ([`3f90666`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/3f90666eb983cf6ac016e45deecd3445ed14b067))
- sync OpenRouter model descriptions and add per-model opt-outs ([`f813d73`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/f813d7339f848e22b0434f37678104c5f34aab0d))
- modularize into package. add timing instrumentation ([`87c777b`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/87c777b0618999c7fec81e35509ea975588270f6))
- enable direct uploads by default with blocklist for incompatible models ([`276e224`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/276e224e885d57d0e672e693826d735417aad1f6))
- assemble per-message session log zips and capture full OpenRouter I/O ([`4020399`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/40203993bd6ef8d78242775277a729e27a0f15c6))
- default new OpenRouter models to admins-only access ([`69f116c`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/69f116c6cfa3d9b39f3ff18a74bedcf810a1d434))
- add model variants support (v2.0.3) ([`530dd68`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/530dd68f051231c24cf748647b288cdbde671699))
- add OpenRouter presets support ([`2d1d5a8`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/2d1d5a8c80722e33410d0733c4975e02bc56c705))
- add OpenRouter presets support ([`1482759`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/14827592329450412578e236b8341c7b235bd10b))
- add OpenRouter presets support ([`878eff8`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/878eff81f77c612a687f4c57d2754bdaedca6136))
- add bundler script and CI workflow ([`94cf6f9`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/94cf6f91c4b2c6f092f879b5cd7ae55162301f86))
- build bundled artifact on all branches ([`446fcf6`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/446fcf604601b59478a2213a23121d9a3e2a6a48))
- add tag-based releases and auto-updating latest release ([`d9f3200`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/d9f32006563c5b7c76da363caf37f2af7f7f3657))

### 🐛 Bug Fixes

- fix model timed model refresh bug ([`bd2427a`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/bd2427acfc5f241021ec2970ddb81041aa943cba))
- Fix logging queue threading and tool args ([`6c8bb2b`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/6c8bb2b27fc4877f32e60a4fb7d2b41a31121052))
- Display reasoning progress and thinking status during extended reasoning ([`ad087a7`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/ad087a776e2159a8652b47d5503391c211aca13c))
- Reduce reasoning status update frequency to prevent UI spam ([`d036d06`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/d036d06e32cc1e774f9d754864f0d85ac57231ea))
- Fix test infrastructure: Make all security tests pass (28/28 PASSING) ([`e9a6eab`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/e9a6eab72d320042352dd84a54bba03025670c6d))
- Fix missing Tuple import in type hints ([`13f8bf8`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/13f8bf89a80873c26d39a407908cecbd92af4285))
- Fix Pylance closure variable errors by converting transform_messages_to_input() to instance method ([`0ca58f6`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/0ca58f6de8122973e71173df943023759171e9ea))
- Fix test_transform_messages.py for instance method change ([`bfa3240`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/bfa324088e6cddc5fbfa219397a90d5effa318a3))
- Fix strict schema required list ([`c0f1aea`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/c0f1aeacffe7f516409711d64f4d6cbaa79f916e))
- Fix user valve log level inherit handling ([`9ca6749`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/9ca67496e2e0a23f4104d732f246c716a298da5e))
- preserve responses parameters ([`c3db081`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/c3db081679126f54bdada95d77cd0253219afd80))
- keep optional tool schema fields optional ([`c62a410`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/c62a410816a5b672bd2e7aa64e10d952c6badac2))
- run model metadata queries in threadpool ([`71165f2`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/71165f2bc66ceeae2872094d78e81888df608930))
- Fix strict schema handling and add tests ([`cfa8bc2`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/cfa8bc20aa8d04b7e6f3a979a906eee305996bb1))
- Fix citation rendering in streaming responses ([`cbf7fd3`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/cbf7fd384fc11a3ecf8ceb0d496319735cd9de9b))
- Fix image inlining guard and base64 encoder ([`62cd889`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/62cd88931f583012c4178f51b6d4a2944d5a0659))
- trust valve bounds ([`613065c`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/613065c4030955a5e3488836fe535b58ff1092e1))
- Fix OPENROUTER_ERROR_TEMPLATE valve description placeholders ([`49876fd`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/49876fda8c32ba491d8eedb87157920ef20fc3d3))
- Fix error template rendering to show all fields conditionally ([`86a088f`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/86a088ff46fa590992756b944a5cfc7922e6a006))
- Fix handlebars conditionals in error templates ([`566e7da`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/566e7da1c7483b49ebc73d7f554369cf06aaa9d9))
- Fix pipe identifier and drop user log level valve ([`a1bfdc4`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/a1bfdc4d837c590ffbf58445f2e626e76b60d245))
- auto-retry gemini reasoning error ([`25f01c7`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/25f01c77e4292e0f040498649e2a94253f35e703))
- reject headerless artifact payloads ([`4b3c57e`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/4b3c57ecfcd8847f9cceb124669b3f59d28c0b7d))
- enforce valve default types ([`3ccbad8`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/3ccbad82d7a3b864bd82c057d8e0793347a79c77))
- harden pipe runtime for pylance stability ([`c17bf6a`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/c17bf6a3bad1861f5b0cddd3f306772d4662e01a))
- Fix Pyright warnings with stricter schemas/tests ([`c8e30f8`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/c8e30f8ec09f3a6ce7ffb68ceff476ad84400abd))
- prevent streaming deadlock with unbounded queues and monitoring ([`40de6a4`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/40de6a4ad9d5251e22a6c6632b5fe0734cc696f6))
- ensure byte type consistency in SSE event parsing ([`79a0652`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/79a0652727d23e3514ab021fcde2d4e9dbeeb16d))
- add defensive type inference to strict schema transformation ([`b586f05`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/b586f05e8e5068fc003cfc5b9f232f6405272707))
- prevent duplicate reasoning blocks in OWUI ([`4e18af7`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/4e18af730ffc6f6540f00506e5d91869b97e413e))
- avoid None.get crashes ([`b0d1261`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/b0d12612ce595bc173a05a1dfff0d5182c0264c1))
- preserve system/developer messages verbatim ([`ab75be7`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/ab75be72ecf70202074b2eb8c3617e9ebb092011))
- harden session log capture against exceptions ([`af69cd1`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/af69cd1b45ee9baae654d97f7814a83352489a37))
- trust validated valves ([`38f3beb`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/38f3beb7cd619f3846eefd28f484fb8317dcfacc))
- enable interleaved thinking streaming ([`59a338d`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/59a338db4dfdbce83be3ca8fc4f37066160fe733))
- clamp tool output status for OpenRouter ([`6d4e573`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/6d4e573dd71ca8961f992b84cde2d463dc440c6e))
- dedupe reasoning snapshots and tolerate invalid tool args ([`5427fb8`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/5427fb8f048a7473722f438a19309e3d2471b0e0))
- make SessionLogger thread-safe ([`21d1bb5`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/21d1bb53f3bb30600bf9460215394dbf5a975231))
- validate redis flush lock release ([`ecae83c`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/ecae83c9653a1dd5b172f208d607cc0112be06ea))
- bound middleware stream queue ([`d82b325`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/d82b325c6f48cf73a376b43389b4244747e45a42))
- log tool exceptions with tracebacks ([`36e6a59`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/36e6a59ae5b63265f525f4e705a2874b11c1bf88))
- surface remote file download failures ([`67bb884`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/67bb884c6f136a47d05bcacdb0fb516266b17407))
- avoid hanging DB executor shutdown ([`840467b`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/840467b6f201f99751256199ef23bf412637913f))
- log redis close failures ([`6794d1a`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/6794d1ac703e46e75b9b5d6a9ae3eda8052de815))
- bound tool shutdown ([`42f3549`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/42f3549d4ec115cf557ee47cf393da8a6aca7391))
- apply Claude prompt caching to normalized ids ([`c67ce78`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/c67ce78e5e453924597d60b96d3148672560fd1b))
- make FORCE_CHAT_COMPLETIONS_MODELS accept slash globs (anthropic/*) ([`2eb1d52`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/2eb1d522fd98c816ec5733956e4187f5b0f27a26))
- translate structured outputs for /responses ([`699f97e`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/699f97ed3f03cf839b530a96498c3ad1273bb7d3))
- harden shutdown/cleanup and bump v1.0.18 ([`e17ddc5`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/e17ddc5caaa7e996263406080a7f8889cb95ebec))
- send identifier valves for task model requests ([`2d57271`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/2d57271f2b994d4fc8d1b667a7d5dbac8aded40c))
- drop uninlineable OWUI image URLs ([`dde50aa`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/dde50aa8ea43f679a0ba3b73ada373cfd9a17ebb))
- honour OWUI flat metadata.features flags ([`b03ff3d`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/b03ff3d702d67f29e452fcf3d0651897dbfbee06))
- always enable file uploads for pipe models ([`dbd563d`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/dbd563df8b049f518e5677b24ec39e1b3c49057c))
- allow install on Python 3.11 ([`aaec9d7`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/aaec9d740ef1c8e0a1f304adbc237d4208f1cc06))
- fail fast on undecryptable OpenRouter API key ([`d0bb45e`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/d0bb45e697bb4a3181fc3c8e436236cec3e49170))
- honor Direct Uploads /responses audio allowlist ([`c9d56ea`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/c9d56ea25fb0626c8377127e746997f02d4b3bc9))
- harden Direct Uploads metadata safety and improve fail-open UX ([`0b62333`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/0b623331f821f06131d777e561120d4aeaf094ae))
- use OWUI metadata tool registry for native tools ([`fb1cc8b`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/fb1cc8b2f172aac70934595a33071efb7fd9eaaf))
- bypass OWUI File Context for diverted direct uploads ([`51c9c31`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/51c9c315903da18c63a091f3f77853e96a84028f))
- harden streaming and breaker edge cases ([`25e81d2`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/25e81d2389ef243a1c45647f8744a59bdadde099))
- use aclose() for async Redis client cleanup ([`6c6c0d3`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/6c6c0d391da5b4fadcc8ce6970008672a06ecfe9))
- default to HTTPS-only remote fetches ([`05438e2`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/05438e2b3bb39a1a9c00ca3a56484d6ef56af094))
- resolve all pyright type checking errors (283→0) ([`e8b83b6`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/e8b83b6d35b93788957b34ad4e05ae6c9c075fb3))
- preserve preset field through responses→chat conversion ([`74b7cf7`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/74b7cf710d88dcdf8d12182bd6bf63f99883838f))
- add permissions for release creation ([`9b8a969`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/9b8a9699cfd261045c4b3c987279d391d2ecf330))
- add permissions for release creation ([`75a971d`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/75a971d351ed5996c496d4428ba792a13685140c))

### 💼 Other

- Initial commit ([`f27cc0f`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/f27cc0f6f418218f184ad5ccdd36efa73e131e39))
- Revise README for OpenRouter Responses API plugin ([`5a1a7a0`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/5a1a7a0d98707d7e6c95001ca6524dac40a751c6))
- initial commit ([`df01048`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/df0104811b003be597defada3e16d9e7da6d5960))
- Update issue templates ([`515020c`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/515020cdfd8554743eb65f86048f740f8a285ca8))
- Update README.md ([`ce2f6e8`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/ce2f6e88d39b7a74df59a3f10119b8bc6f7d38bc))
- Propagate cancellations to OpenRouter pipe ([`f432c83`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/f432c83dcf14c018036c6ba5a296346d7cd2cf0f))
- Enhance README ([`1561362`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/15613621f3b27cf0ee3594be9d4296be76d6b3ec))
- update gitignore ([`0c4abb7`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/0c4abb72cf297266489ab25f303f4dd75ffd6671))
- Add timeout to Redis ping warmup ([`dc76753`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/dc76753b11622af9feaf70f6370a9f16f62cbe55))
- Add turn-aware tool output pruning ([`fa3cb7a`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/fa3cb7ae84e6b6e124900bec42c5264d69dfb275))
- Clarify tool-turn retention doc ([`68b9d32`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/68b9d3245ef70bfa4aa781f1fcab4e01d15faa52))
- Allow pipe-id fallback from request model ([`167c75e`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/167c75e45d06edf9186ccc47ac6bdd5f2861e698))
- Guard pipe request future failures ([`d7cc465`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/d7cc465c039d60227fc2a6b56aa8c6bb3e68ae81))
- Gracefully handle catalog refresh errors ([`b99c1d1`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/b99c1d1aec0f5b1ad845a561b73f64f3eefcf9b6))
- Provide fallback pipe identifier ([`d37f802`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/d37f8024d3311e8d3d249e47e7d345206d25da2f))
- Handle tool registry load failures ([`a86f02a`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/a86f02ab84f6df7613ff03fc63e5460b437a90dd))
- Harden model function-calling toggle ([`c9d6ae3`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/c9d6ae311a29253f95e20cac43b558cb7e4c5b69))
- Guard citation persistence failures ([`d7ed9d9`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/d7ed9d95fff5e11533dbe128b285beb1695634b4))
- Wrap event emitters to survive disconnects ([`60949eb`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/60949eb64e0b73ba3ae7d2e077f5d4f3df40f20b))
- Remove duplicate CSS injection log ([`089504f`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/089504f192e105a04003aca2859c65f2b076fda1))
- Add guard regression tests ([`84f9910`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/84f9910421b99d3230a3d06241221d6869e4f088))
- Add 'Model' field to bug report template ([`d8517af`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/d8517af54a5e8f47b312415a0d58133112087d13))
- Guard replaying orphaned tool artifacts ([`d4b8672`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/d4b8672d2a754a773f32e8bde6c705f3e2a19357))
- Stabilize pytest bootstrap and fix Pydantic compat ([`1722f30`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/1722f30f8a30db6f15b041d62818dd97f90dc88e))
- improve reasoning stream + stop crashing on emojis ([`d76455f`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/d76455ff328b9fef3a431b706fa74d20d4ef7860))
- Enhance streaming controls and defer OpenRouter warmup until configured ([`32196e3`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/32196e329c3eaa6651e05eaf04c8e33fdc752f82))
- Ignore venv and pytest cache ([`79aad6d`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/79aad6d1bb5a2fe3dcfb7326b2c018c783049aa3))
- Add tuning valves for streaming, tools, and redis ([`ec57076`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/ec57076d1fdb65f035d03995726c167ab7abce75))
- Normalize docs and tests line endings ([`f8ebe47`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/f8ebe47d9a45f443bca0738153f21b5a520c1b98))
- Improve reasoning progress display with semantic chunking ([`362d870`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/362d8706278d66e6b62b0ce17aa697cf95be8ec3))
- Add IDE and tooling directories to gitignore ([`e05c26b`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/e05c26b3dcc89bcd89ac62e6c8eee327118a59d6))
- Replace image link in README ([`5e91705`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/5e91705d97a3d02e4b4416562dcca449d0e3c3bb))
- Add comprehensive multimodal input support with retry logic ([`4bfe04c`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/4bfe04c0b3280b7ca978e1f88338fdd5ec8d0f4b))
- Add production-ready configurable size limits for multimodal inputs ([`b0218a4`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/b0218a46acb8f67c0d87aa5653ead37fbe66a54c))
- Add audio file MIME type detection for automatic format conversion ([`130bbfa`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/130bbfa4a5277c768c74feaa047d688869082408))
- Add comprehensive multimodal support improvements ([`3504462`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/3504462ee38a4c770227c8287483563613d7e728))
- Add critical security fixes and improvements ([`688af47`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/688af47e800984ce9de638a9cd25bf0d7f2c5f9b))
- Add passing unit tests for security fixes ([`41bc95f`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/41bc95fdd2200c7e4a2cbfad018c0de12b4236e2))
- Add skip marker for test_security_fixes.py due to Pydantic compatibility ([`c608b84`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/c608b8440e655910c6c170567a42e1c479529a6e))
- Remove obsolete limits field check for max_completion_tokens ([`00dfe7b`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/00dfe7b3e13687c9e2905ad7f75a697f662f046b))
- Update SessionLogger console format to match loguru style ([`01c4b66`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/01c4b6635678a1abbb0d16565cd3849665c76aa6))
- Normalize line endings from CRLF to LF across all files ([`9191585`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/91915851bdb5dd6b87a64168db76aa8c65db1f77))
- Align remote download caps with RAG settings ([`694e07e`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/694e07ecd51ad2dd407f4ee6e1ef62dd97bde76a))
- Refresh README overview and feature summary ([`c1406c8`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/c1406c8068f1281f7c88ca9962eae0c2373f84b2))
- Plumb request/user context into multimodal transformers ([`13ca1f3`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/13ca1f3d9da3f417ad55d228825961c3ec0932c3))
- Host remote file URLs in OWUI storage ([`f874150`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/f8741505226ab8153126d3893c815c077666541c))
- Add valve for remote file_url rehosting ([`c0a6d5f`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/c0a6d5f19865aa42e2406174ab35ceff918b63dd))
- Harden audio inputs and fix request context ([`012f23d`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/012f23de9dfd2348714cd157a128192cb87846c5))
- Ensure multimodal uploads always rehost ([`0875573`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/087557311af4a4b6403d20db485c1485dec6676c))
- Valve fallback storage identity ([`8ea30a4`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/8ea30a4255cb36b7413c1b60b4c1e5b04e9185a4))
- Filter remote download retries ([`6be5617`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/6be561720336f4875a3256e0d808fb6993f973fc))
- Honor retry-after and 425 for remote downloads ([`7f3dd0d`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/7f3dd0dc0e190ed6597244a2f0bc5a1f9aea15b0))
- Improve remote download safety and fallback storage ([`9651eb3`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/9651eb30f045bb3dec8158e9cbc6413d48081c29))
- Clarify encryption requirement in audit doc ([`cb39db1`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/cb39db1b61be549b882dc24ada937bdb11272877))
- Default to always attempt compression ([`5c9d61f`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/5c9d61f816a1691461838e6850b810f426116924))
- Preserve optional fields in strict tool schemas ([`5f79244`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/5f79244870c494872ae06c411224fd5c953506e1))
- Run SSRF validation asynchronously ([`7d81b95`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/7d81b95f2e697de435aa923a84aa2b45900c4dbe))
- Honor user valve overrides without clobbering defaults ([`1a232ed`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/1a232edee22ab37ce089c7adf06c2a4ba31e9347))
- require owui multi-worker settings before enabling cache ([`77313bc`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/77313bcc6e73ee8da270eda56148dc43331361b0))
- Add regression coverage for pipe helpers ([`6a02c7a`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/6a02c7a0a17b8d6eae6987b99fa4427f16288c47))
- Add OpenRouter error template docs and streaming fixes ([`6908299`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/69082995aa700e64cb211a777845350bdbd6a2f6))
- Refine OpenRouter error templating ([`97527a2`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/97527a2295f468f1df7ea861713d4fc91d4a5da5))
- Emit status on provider errors ([`f532f8f`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/f532f8f6ee8691d8df5a57ca1f928cbb8ac4e009))
- Improve provider error handling ([`7d59ca0`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/7d59ca0422ec1aab8accfd32d2ef6107b8ab2cf8))
- Enhance documentation with example template ([`0f39d4c`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/0f39d4c5041c291f84c6646fd34bab52ebea8aaa))
- Add packaging metadata and clean up docs ([`5c2bd2d`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/5c2bd2dab36d7e7cd005dd77e1c6fbd5a00f4350))
- Update project title in README.md ([`d4faa9c`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/d4faa9cb255d31f4f2620f3d294313a84ae07658))
- Restore OpenRouter X-Title header ([`05a3ca4`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/05a3ca4aeb1515f6c2520674eb2004737a959075))
- Ignore vendored and egg-info artifacts ([`2151d3a`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/2151d3a720ee1a5a87d637bbe3558d4366d50756))
- Harden fallback storage user creation ([`57cc8ed`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/57cc8ed75c00aa6bbcaea1c1a72a3b838e5a5895))
- Add docstrings to internal helper functions and methods ([`1af8ab3`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/1af8ab352ca2515f80a9aeebb89576e29e6a511e))
- Implement comprehensive error template system for all error types ([`dc8a566`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/dc8a566e18df4c49b2592dcb6f72256342fc16d2))
- Improve OpenRouter error handling ([`c9d1e93`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/c9d1e933851ff94c7d604d6ae0725385dfb1437a))
- Ignore local planning artifacts ([`a29ec77`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/a29ec77b332b7bc1c55b6ce3ec84abd5d74a9a96))
- Refine streaming behavior and metadata ([`75d98ca`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/75d98ca176ae8c1a41be9447c2bcf5a4fdbbdd54))
- Rename pipe to Open_WebUI_OpenRouter_pipe ([`400c9ba`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/400c9ba6927226deb8447ccbb9c57a6f812a49e8))
- Update README version badge ([`2b8dfdb`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/2b8dfdbf6eeea915a0f4b99df4ccd7bb220912f5))
- Improve task task routing and reasoning controls ([`59eda90`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/59eda9052f13226449b9c34c619f7a639fc6d499))
- Clarify user valve labels and descriptions ([`38ff069`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/38ff069bfabda6f09e78a5938a52d19582e72c06))
- Drop unused user-valve alias plumbing ([`8ca272a`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/8ca272a45a43d9ac3fa74a8c8ff6744d40a51bc5))
- Add optional Redis cost snapshots ([`3fa9e50`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/3fa9e501282c9ebc11ca2acb5df6bc877050b9a9))
- Map legacy function_call to tool_choice ([`ae40aef`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/ae40aef92741aee33eedc48886b9bdff0a720144))
- Add breaker valve tuning ([`11b8e82`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/11b8e8295f5b8206e6080ffa4dab766aeb568ff9))
- Capture task model cost snapshots ([`3ea619a`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/3ea619a893965a6e0743925b21c3f47018d362ce))
- Clarify ENCRYPT_ALL defaults in security guide ([`53748b0`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/53748b0f3a91ce0d16f83c71b5c5ec36875afd67))
- Support optional emitters ([`8969331`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/89693314ff84166b32a15d49367a78a1d4c61ff0))
- Stop replaying non-replayable artifacts ([`6ddf8a0`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/6ddf8a0a1bb81aafc7f1ce16216da83238e25920))
- Update issue templates ([`f40e35d`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/f40e35d300cefa99ccdc031b68c732d2e7b4c0a3))
- Update openrouter_integrations_and_telemetry.md ([`6a58ac7`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/6a58ac702f76831a3d5567a654aae64986dd174c))
- Revise documentation relationship map format ([`1fe678e`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/1fe678e32e7d1ce83bf053a7ecedf7f6528bba97))
- Delete work files ([`1bf7148`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/1bf7148d7741738c0c5587afed479f81e39f31f7))
- Update QUEUE_MAXSIZE ([`a750591`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/a7505912ee8ac0428722f6eeaaa43e59d65b990d))
- Update readme and venv script ([`db15f23`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/db15f23d7c183ea55b65e94138ecf55919557a7c))

### 🚜 Refactor

- Refactor and update default timeouts ([`41c9741`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/41c97411f9b649241ac9551cf093741da9a88552))
- Modernize code with Pydantic v2 and improved type safety ([`aa9dab3`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/aa9dab34e675488802d581ff42da72136d21f9ee))
- move message transformer to pipe ([`009f3fb`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/009f3fb6e608ca0aec2f1407afcc2a75a9cda57f))
- Refactor documentation for clarity and conciseness ([`ef260dd`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/ef260dd2a0f670cf30a3a3e8c9bb4a55f94a6974))
- make OWUI file handling file_id-first ([`f7b85e3`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/f7b85e34f9f2a69bcc43c12f985296ead308f753))

### 📚 Documentation

- Document architecture and normalize logging ([`7e4d502`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/7e4d50258b155d5c5796ebdbacb25da4c17bf34d))
- Document system and user valves ([`ce07e0b`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/ce07e0bbe4589f08b38f3bce7c3f8e027bd60a76))
- Document artifact persistence pipeline ([`ff1701d`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/ff1701d8fd20f1ac433e609b7f1d8ec9a47bab72))
- note pytest PYTHONPATH requirement ([`961bf05`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/961bf05917e8ace84a5b93b0a07c980bd1a9a8d1))
- call out preserved responses parameters ([`906d499`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/906d49918a320c6f99c6581d1fb33e8d1fccc692))
- refresh production readiness audit ([`305e743`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/305e7434efd7503df0c445ad1294fc61410a5667))
- note breaker self-healing ([`90e6c6d`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/90e6c6d0fb425d5850b5a8f75f2a1eed01d20097))
- comprehensive security guide and documentation accuracy improvements ([`becf111`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/becf111a9a6866cdc273f1c02d3e00b21244d5d6))
- Document expanded OpenRouter error templates ([`ac11bc6`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/ac11bc61cdd24316f88d94bf7b0961e741bdb016))
- Document cost telemetry and Deep Research statuses ([`96b0f78`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/96b0f78d48a8812fb6173a314bd2a56f1a1d63cb))
- Document legacy function_call mapping ([`7be78a1`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/7be78a1f0c89881a97807804c827a1f00f26db3f))
- add task model guidance ([`84f19b1`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/84f19b1ab17a1ad7f5cf98e464beb99511527f5b))
- Document recent changelog entries ([`bdd9ed0`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/bdd9ed0106200530e6681380419634c89d87c160))
- enhance navigation with persona-based paths and cross-references ([`b23c9c8`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/b23c9c8df2aa38f581634d2a6a0a10d3326f6144))
- convert file references to clickable links in documentation index ([`3d73504`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/3d73504b991caf2270af7163a5033346742cca6d))
- fix README screenshot ([`dac4211`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/dac42116a80447d20ea40d22081293ab896bc8ee))
- fix documentation index section placement ([`1eb89e4`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/1eb89e4d5e919d3d3a9129ac5891ac36fcda5f19))
- link session log storage deep-dive ([`620bf9c`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/620bf9c2203e120d1ee1a6403f402e2a5a75e474))
- update changelog ([`e33a82d`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/e33a82d3536df5b260f86bd96531e641e4e47c60))
- refresh docs and bump version to 1.0.12 ([`5c6363a`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/5c6363a75ea3e104c17d629cf8fea83d1470aaf0))
- document interleaved thinking valve ([`09be5cf`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/09be5cf130ab12182f83c09689d881d1a2ee8f50))
- align Python version with Open WebUI ([`042c7aa`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/042c7aada0651ec4c31ee61e4d5c361d1291b903))
- explain Direct Uploads vs OWUI File Context ([`6307d87`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/6307d87af89c70f3912cf58fe4e29ae91c7576b6))
- clarify stub auto-update and fork behavior ([`8c3db36`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/8c3db36b060b52fb66ad917c457f3ff331b7e312))
- note OpenRouter preset field issue in testing ([`1631f81`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/1631f81c0ca6a03ba55aae3aa3f376c54daeff44))

### 🧪 Testing

- align remote download mocks with streaming ([`bb89a2b`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/bb89a2be4abb2884feb845a53c54dd499ed550bd))
- add image transformer coverage ([`65ea555`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/65ea5558fe8ec3770f2dc4a915fb6a8424ce5967))
- exercise real ssrf and youtube helpers ([`ddb1519`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/ddb1519f99128542deb6948b705d96e507b9f0da))
- tighten transform coverage ([`3bbe9e8`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/3bbe9e8d4741e448f99fcd1b66e66b7a81ca4fe9))
- expand coverage for streaming and helper utilities ([`d8d34b3`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/d8d34b3dc1a3cc4866e9c47948bec2153c4a4a8f))
- set stream=true in streaming loop tests ([`9b13274`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/9b13274152186b0cb8d40177d6ec1c1515c72189))
- strengthen coverage and reduce drift ([`ba500a5`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/ba500a5259467d16dc7ef72e630ade89ae5c50da))
- consolidate test suite and fix async cleanup issues ([`d67234e`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/d67234ea7119de3d4527b7bf585743087a94ce11))
- tolerate extra timing events in logger tests ([`3159cb2`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/3159cb27e586a4d03ff6c96a19f3ac8184c8c673))

### ⚙️ Miscellaneous Tasks

- remove unused helpers ([`bc756f2`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/bc756f2e8a556f86b87659fdf50ca7667130cbce))
- remove duplicate ResponsesBody debug log ([`e188c2f`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/e188c2fd6fbaeefca871440788dcd06ca2d1dc12))
- dead-code cleanup and pipe id simplification ([`4b252c6`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/4b252c6b359630455b47d91b674f07dfd1ce3fe6))
- remove redundant valve casts ([`ab5d7c4`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/ab5d7c4b2abdd3304b9a7e95ab253bf712f8825e))
- exclude backups from pytest discovery ([`6168d48`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/6168d48c27ffcf2c5c87b32a0fc063775840e6c7))
- add dev env repro + usage ingest helper ([`9bb41e2`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/9bb41e29506d7f9378b27bbc7d06dd48290bf1cb))
- redact data-url base64 blobs in debug logs ([`353b123`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/353b123665ea9136ad1254c718b651901c1b22e1))
- checkpoint 1.1.1 ([`97bd2b0`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/97bd2b034f6b3146fe2e10f99eb420b084ef7a53))
- consolidate shared helpers and constants ([`6ab3872`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/6ab3872817e6f1408f41653592ca14d2e8c441b9))
- add GitHub Actions workflow with venv + deps ([`c38f8df`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/c38f8df97c693ce3e33811585410d412f54552fa))
- clean up aiohttp ClientSession leaks in tests ([`f4f35aa`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/f4f35aa88d1b98a625e09d4e40ac895cf84bfda0))
- install aioresponses for tests ([`3eaee84`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/3eaee84805ed67652da6635b8117b2b51ca55639))
- trigger bundle workflow ([`f2efba6`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/f2efba6b0be82f0dbe983a63ccaaacc8b43261ca))

### ◀️ Revert

- Revert "Add audio file MIME type detection for automatic format conversion" ([`93f0daf`](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/commit/93f0dafbe114a6c6cac63f8b0d551db9d99750c7))

