# Release notes pending

## Behaviour changes

- **Streamed replies to programs with no chat are built by Open WebUI's own chunk helper again on Open WebUI 0.13.0** - Open WebUI 0.13.0 renamed the helper (`openai_chat_chunk_message_template` is now `chat_completion_chunk_template`), so the pipe fell back to its own copy, whose chunk ids look different and carry no `logprobs` key. The pipe now uses the new name, the old one on Open WebUI 0.12, and its own copy only when neither exists.
