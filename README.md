# Open WebUI → OpenRouter Pipe

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Version](https://img.shields.io/badge/version-2.0.1-blue.svg)](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe)
[![Open WebUI Compatible](https://img.shields.io/badge/Open%20WebUI-0.6.28%2B-green.svg)](https://openwebui.com/)

**Access 300+ AI models through one beautiful interface.**

Use GPT-5.2, Gemini 3, Claude Opus, Llama 4, and hundreds more — all from your Open WebUI, all through OpenRouter's unified API.

![output](https://github.com/user-attachments/assets/c937443b-f1be-4091-9555-b49789f16a97)

---

## What You Get

🎯 **Every Model, One Place**
GPT-5.2, Gemini 3, Claude Opus, Llama 4, DeepSeek, Qwen, Command R+ — browse them all, try them all, compare them all. One API key, one bill.

🖼️ **Multimodal That Actually Works**
Drop in images, PDFs, documents. The pipe figures out what each model supports and handles the rest.

🔧 **Tools & Web Search**
Your Open WebUI tools work seamlessly. OpenRouter's native web search is one toggle away.

🎨 **Beautiful Integration**
Model icons and descriptions sync automatically. Capabilities show up in the UI. It feels native because it is.

💬 **Clear Communication**
Helpful error messages, real-time status updates, and transparent cost tracking per request.

---

## For IT & Operations

⚡ **Production Hardened**
Rate limiting, circuit breakers, request admission controls, and graceful degradation — built for real workloads, not demos.

🔐 **Security First**
Encrypted credential storage, SSRF protection on remote fetches, no secrets in logs. Designed for environments where security reviews happen.

📊 **Cost & Attribution**
Track spending per user, per session. Optional Redis export for billing integration. Know who's using what.

📝 **Audit Trail**
Optional encrypted session logs for incident response. Request identifiers flow through to OpenRouter for end-to-end attribution.

🛡️ **Enterprise Controls**
Encryption, retention policies, request attribution, and operational hooks your governance program can build on.

---

## Quick Start

**1. Install**

In Open WebUI: **Admin Panel** → **Functions** → **+** → **Import from Link**

```
https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/raw/refs/heads/v2.0.1/open_webui_openrouter_pipe.py
```

**2. Enable**

Toggle the pipe **ON** (the switch next to the function name).

**3. Add Your API Key**

Click the **⚙️ gear icon** on the pipe → paste your [OpenRouter API key](https://openrouter.ai/keys) → **Save**.

**4. Select a Model**

Back in the chat, click the model dropdown — you'll see all OpenRouter models. Pick one.

**5. Chat!**

That's it. Start talking.

---

## Requirements

- Open WebUI 0.6.28+
- An [OpenRouter](https://openrouter.ai/) account
- `WEBUI_SECRET_KEY` configured (required for encrypted credential storage)

---

## Documentation

Everything else lives in [`docs/`](docs/README.md):

- [Configuration Reference](docs/valves_and_configuration_atlas.md) — all the knobs and switches
- [Security Guide](docs/security_and_encryption.md) — production hardening
- [Tool Integration](docs/tooling_and_integrations.md) — extending with tools
- [Cost & Attribution](docs/openrouter_integrations_and_telemetry.md) — billing and tracking
- [Troubleshooting](docs/error_handling_and_user_experience.md) — when things go sideways

---

## License

MIT — use it, fork it, ship it.
