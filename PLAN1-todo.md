# PLAN1 TODO (OWUI 0.7.x upgrade leftovers)

This list is derived from `possible_todo/*.md` and a quick check of the current pipe code/tests.
It focuses on what still looks **unfinished / not re-verified**, not what’s already implemented.

## Outstanding

- [ ] **Decide policy for OWUI builtin tools passthrough**
  - The pipe can now pull OWUI builtin tools via `open_webui.utils.tools.get_builtin_tools(...)` and advertise them upstream.
  - Open question: do we always want OWUI builtin tools (time/chat/etc.) sent to OpenRouter, or should this be valve-gated to avoid schema bloat?

- [ ] **Add a focused regression test for OWUI internal file URL generation**
  - We previously had OWUI route ambiguity risk (`get_file_content_by_id` duplicate endpoint names in OWUI 0.7.x).
  - Add a unit test that asserts we don’t rely on the ambiguous route name (or that our fallback behavior stays stable).

- [ ] **Re-validate “native function calling” behavior in a real OWUI 0.7.x deployment**
  - OWUI 0.7.x changes behavior when `function_calling == "native"` (skips memory/web_search/image_gen/knowledge injectors).
  - Run an ops smoke checklist in OWUI (native on/off) for:
    - memory enabled
    - web search enabled
    - image generation enabled
    - knowledge attached

- [ ] **Clean up stale `possible_todo` references (optional but reduces confusion)**
  - `possible_todo/plan1-codex-to-check.md` references a `tests_owui_head/` suite that is not present in this repo.
  - `possible_todo/TASK1-CONCLUSION.md` references `tests/test_builtin_tools_interaction.py` which is not present.
  - Several line-number anchors in `possible_todo/PLAN1.md` / `possible_todo/PLAN2.md` no longer match current code.

- [ ] **(Optional) Add back an OWUI HEAD “source verification” suite**
  - If we still want automated drift checks against `.external/open-webui` (AST/regex-based), reintroduce a lightweight `tests_owui_head/` suite as described in `possible_todo/plan1-codex-to-check.md`.

## References

- `possible_todo/PLAN1.md`
- `possible_todo/PLAN2.md`
- `possible_todo/plan-codex.md`
- `possible_todo/plan1-codex-to-check.md`
- `possible_todo/TASK1-CONCLUSION.md`

