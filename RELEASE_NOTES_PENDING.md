# Release notes pending

## Behaviour changes

- **Works with Open WebUI 0.13.5** - Open WebUI 0.13.5 renamed Terminals to Machines, and with it the pieces the pipe uses to show Open Terminal files in a chat. When the pipe ran the tools (Pipeline mode, the default), a file the model wrote or showed through Open Terminal no longer refreshed the chat's file browser, opened the preview panel or kept its file card, and a person's choice to show those files inline, made after the upgrade under its new name (Machine File Display), was ignored. In Open-WebUI tool mode, when Open WebUI asked the model to finish after a tool limit (too many tool calls, or the same call failing repeatedly), the pipe hid the results the model had gathered in that reply, so the final answer could not use them; the model now gets them. All of this works on Open WebUI 0.12.0 to 0.13.5.
