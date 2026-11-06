# Release notes pending

## Behaviour changes

- **Streamed replies to programs with no chat are built by Open WebUI's own chunk helper again on Open WebUI 0.13.0** - Open WebUI 0.13.0 renamed the helper (`openai_chat_chunk_message_template` is now `chat_completion_chunk_template`), so the pipe fell back to its own copy, whose chunk ids look different and carry no `logprobs` key. The pipe now uses the new name, the old one on Open WebUI 0.12, and its own copy only when neither exists.
- **On Open WebUI 0.13.0 an automatic update of the pipe is announced to event functions and webhooks again** - Open WebUI 0.13.0 publishes an event only for a request, and the dashboard's automatic update runs without one, so its `function.updated` event was dropped and the failure was logged only at debug level. The update now hands Open WebUI a request built around its app, the way Open WebUI announces its own start-up, and an announcement Open WebUI does not take is logged as a warning, once.
