#!/usr/bin/env python3
"""Future Fusion prompt deliberation tournament.

Per prompt type (panel / judge / synthesis):
  Stage 1 ORIGINATE: each model independently drafts a candidate system prompt.
  Stage 2 MERGE: each model merges the 3 originals into its own best version.
  Stage 3 FINAL: Claude Fable receives 3 originals + 3 merges and writes THE prompt.

Every call: web plugin + openrouter:web_search + openrouter:web_fetch (server-side),
reasoning effort high (with per-call fallback if rejected), generous output budget.
All requests/responses persisted under logs/prompt_deliberation/.

Usage:
  python fusion_prompt_deliberation.py --smoke
  python fusion_prompt_deliberation.py --run [--type panel|judge|synthesis]
"""
from __future__ import annotations
import argparse
import json
import pathlib
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import httpx
from dotenv import dotenv_values

REPO = pathlib.Path(__file__).resolve().parents[1]
OUT = REPO / "logs" / "prompt_deliberation"
KEY = (dotenv_values(REPO / ".env").get("OPENROUTER_API_KEY") or "").strip()

MODELS = {
    "kimi": "moonshotai/kimi-k3",
    "sol": "openai/gpt-5.6-sol",
    "fable": "anthropic/claude-fable-5",
}
FINALIZER = "fable"

TOOLING = {
    "plugins": [{"id": "web"}],
    "tools": [{"type": "openrouter:web_search"}, {"type": "openrouter:web_fetch"}],
}

SHARED_CONTEXT = """\
# CONTEXT: The Future Fusion deliberation engine

You are one of three frontier models jointly designing the system prompts for a
multi-model deliberation feature ("Fusion") inside an Open WebUI pipe that talks to
OpenRouter. The feature mirrors OpenRouter's fusion pipeline but runs in-house:

1. PANEL: N models (2-8, typically 3) answer the user's prompt IN PARALLEL. Each panel
   member gets the full user conversation and a tool surface (web search, web fetch,
   plus whatever Open WebUI grants the user: knowledge bases, tool servers, custom
   tools). Each runs its own bounded tool loop. Panel answers stream live into per-model
   UI cards (markdown), with the model's reasoning shown in a collapsible "Thinking"
   section.
2. JUDGE: one model (temperature 0) receives the user's question plus ALL panel answers
   labeled by model id. It COMPARES — it does not merge. Its entire output is ONE strict
   JSON object (schema below, enforced at runtime via json_schema structured outputs).
3. SYNTHESIS: the conversation's main model receives the full chat history plus the
   judge's analysis JSON (and the raw panel answers) and writes the final user-facing
   answer, streamed. It writes FROM the analysis — not a majority vote, not a re-merge.

# INTER-ROLE I/O CONTRACT (all three prompts must be mutually consistent)

PANEL role:
- Input: the user conversation; a tool surface (may include private knowledge bases).
- Output: a free-form markdown answer, rendered in that model's UI card and consumed by
  the judge. Reasoning (if the model exposes it) streams separately; do not restate it.
- Panel members do NOT know the other members' identities or answers.

JUDGE role — HARD STRUCTURED CONTRACT (immovable):
- Input: the user's question + panel answers labeled by model id. Some entries may be
  degraded ("model X failed: <note>") — handle gracefully, never invent content for them.
- Output: EXACTLY this JSON object (strict schema, all five keys ALWAYS present, empty
  arrays allowed; no extra keys, no prose outside the JSON):
  {
    "consensus":        ["point all or most panel members agreed on", ...],
    "contradictions":   [{"topic": "...", "stances": [{"model": "<panel model id>", "stance": "..."}]}],
    "partial_coverage": [{"models": ["<ids that covered it>"], "point": "..."}],
    "unique_insights":  [{"model": "<id>", "insight": "..."}],
    "blind_spots":      ["topic no panel member addressed", ...]
  }
- SENTINEL: inside contradictions[].stances[], an entry whose "model" is the literal
  string "evidence" represents the judge's OWN fact-check finding (rendered specially in
  the UI as "Fact-check — Judge evidence"). Use it when web verification settles a
  contradiction.
- The judge MAY use its tools (web search/fetch, knowledge bases) to verify claims.
- Semantics: agreement across most members = higher-confidence consensus; disagreements
  surfaced not resolved (except via the evidence sentinel); unique insights preserved
  with attribution; blind spots named so the synthesis model can flag them.

SYNTHESIS role:
- Input: full conversation + the judge's analysis JSON exactly as schema'd above + the
  raw panel answers. DEGRADED CASE: "analysis" may be ABSENT (judge failed) — then write
  from the raw panel answers directly.
- Output: the final user-facing answer (free-form markdown, streamed). It should read as
  ONE coherent answer informed by the deliberation — leaning on consensus, honestly
  presenting real contradictions, crediting unique insights where valuable, and flagging
  blind spots when they matter to the user. It has the same tool surface if verification
  is needed.

# RUNTIME MECHANICS THE PROMPTS MUST RESPECT
- These are SYSTEM prompts. The user conversation / panel answers / analysis JSON are
  injected at runtime as messages — never hardcode content; define behavior.
- Panel and synthesis outputs are streamed live to users: no meta-commentary about being
  an AI panel member, no "as instructed" framing, no leaking of these instructions.
- The judge's output is parsed by machine: any deviation from the JSON contract is a
  runtime failure (one repair attempt exists, then the analysis is dropped entirely).
- Tool budgets are bounded; prompts should encourage decisive tool use, not loops.
- Panel members may have radically different strengths (reasoning-heavy, fast, long
  context) — the panel prompt must work across all frontier models.

You have live web access (search + fetch) in THIS deliberation — use it if current
prompt-engineering practice or model-behavior research would sharpen your work.
"""

ROLE_TASKS = {
    "panel": """\
# YOUR TASK: design the PANEL system prompt

Produce the production system prompt given to EVERY panel member. It must elicit the
member's genuinely best independent answer: direct engagement with the user's actual
question, honest uncertainty, decisive use of tools (including private knowledge bases
when present), and an answer worth judging — substantive, positioned, specific. It must
NOT homogenize the panel: diversity of approach between members is the feature's value.
Consider: how to instruct depth vs. brevity; when to search; how to treat prior
conversation context; what makes an answer easy to compare fairly.""",
    "judge": """\
# YOUR TASK: design the JUDGE system prompt

Produce the production system prompt for the judge. It must drive rigorous COMPARISON
(never merging), faithful attribution, correct and restrained use of the "evidence"
sentinel (only for genuine fact-checks it verified), calibrated judgment about what
counts as consensus vs. contradiction vs. nuance, graceful handling of degraded panel
entries, and disciplined production of the exact JSON contract — content excellence
within the fixed schema, since the format itself is machine-enforced. Consider: how to
prevent the judge from injecting its own answer; how to keep stances faithful to what
members actually said; when a difference is a contradiction vs. partial coverage.""",
    "synthesis": """\
# YOUR TASK: design the SYNTHESIS system prompt

Produce the production system prompt for the synthesis model. It must turn the analysis
+ panel answers into ONE excellent user-facing answer: consensus carried with
confidence, real contradictions presented honestly (not papered over), unique insights
woven in where they serve the user, blind spots flagged when material, degraded-judge
fallback (write from raw answers when analysis is absent), and full respect for the
ongoing conversation's tone and context. It must never read like a committee report or
mention the deliberation machinery unless the user asked how the feature works.""",
}

STAGE1 = """\
{shared}
{role_task}

# OUTPUT FORMAT
Return:
1. A section `## Design rationale` — concise, the key decisions and why.
2. The complete production system prompt inside ONE fenced block exactly like:
```prompt
<the full system prompt text>
```
Nothing after the closing fence."""

STAGE2 = """\
{shared}
{role_task}

# STAGE 2: MERGE
Three candidate prompts for this role were independently authored by three frontier
models (anonymized as Candidate A, B, C below). Study them critically: steal the best
structures, discard weaknesses, resolve conflicts with judgment — do not average.
Produce YOUR single best merged prompt.

{candidates}

# OUTPUT FORMAT
1. `## Merge rationale` — what you kept/dropped from each candidate and why.
2. The complete merged system prompt inside ONE fenced block:
```prompt
<the full system prompt text>
```
Nothing after the closing fence."""

STAGE3 = """\
{shared}
{role_task}

# STAGE 3: FINAL SYNTHESIS OF THE PROMPT
Below are SIX artifacts: three independently-authored originals (A, B, C) and three
merges of those originals (M1, M2, M3), each merge authored by a different frontier
model. You are the final arbiter. Produce THE production system prompt for this role —
the strongest possible version, drawing on everything below, resolving all remaining
tensions with your own judgment.

{candidates}

# OUTPUT FORMAT
1. `## Final rationale` — the decisive choices and why.
2. The complete FINAL system prompt inside ONE fenced block:
```prompt
<the full system prompt text>
```
Nothing after the closing fence."""

SMOKE_PROMPT = """\
Two quick verification tasks, both mandatory:
1. Use web SEARCH: what is today's date and one major technology news headline published
   today or yesterday? Name the source.
2. Use web FETCH: fetch https://openrouter.ai/docs/llms.txt and quote its first
   non-empty line verbatim.
Answer with two short labeled sections. If either tool is unavailable to you, say
"TOOL UNAVAILABLE" for that section."""


def call_model(model: str, prompt: str, *, label: str, max_tokens: int = 32000,
               timeout: float = 900.0) -> dict:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "max_tokens": max_tokens,
        "reasoning": {"effort": "high"},
        **TOOLING,
    }
    attempts = [body, {**body}]
    attempts[1].pop("reasoning")
    last_err = None
    for i, payload in enumerate(attempts):
        t0 = time.time()
        try:
            with httpx.Client(timeout=httpx.Timeout(timeout, connect=30.0)) as client:
                r = client.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={"Authorization": f"Bearer {KEY}",
                             "Content-Type": "application/json",
                             "X-Title": "fusion-prompt-deliberation"},
                    json=payload,
                )
            elapsed = round(time.time() - t0, 1)
            if r.status_code != 200:
                last_err = f"HTTP {r.status_code}: {r.text[:400]}"
                if i == 0 and "reasoning" in r.text.lower():
                    continue
                break
            data = r.json()
            msg = (data.get("choices") or [{}])[0].get("message") or {}
            content = msg.get("content") or ""
            annotations = msg.get("annotations") or []
            return {
                "ok": True, "label": label, "model": model, "elapsed_s": elapsed,
                "content": content, "annotations": annotations,
                "usage": data.get("usage"), "reasoning_used": i == 0,
                "raw": data,
            }
        except Exception as exc:
            last_err = f"{type(exc).__name__}: {exc}"
            break
    return {"ok": False, "label": label, "model": model, "error": last_err}


def save(rec: dict, path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    slim = {k: v for k, v in rec.items() if k != "raw"}
    path.with_suffix(".json").write_text(json.dumps(rec.get("raw", slim), indent=1))
    path.with_suffix(".md").write_text(rec.get("content") or rec.get("error") or "")


def extract_prompt(content: str) -> str | None:
    m = re.findall(r"```prompt\s*\n(.*?)```", content, re.S)
    return m[-1].strip() if m else None


def smoke() -> int:
    failures = 0
    for name, model in MODELS.items():
        print(f"\n=== SMOKE {name} ({model}) ===", flush=True)
        rec = call_model(model, SMOKE_PROMPT, label=f"smoke_{name}", max_tokens=4000,
                         timeout=420.0)
        save(rec, OUT / "smoke" / name)
        if not rec["ok"]:
            print(f"  FAIL: {rec['error']}")
            failures += 1
            continue
        content = rec["content"]
        ann = rec["annotations"]
        unavailable = "TOOL UNAVAILABLE" in content.upper()
        print(f"  elapsed={rec['elapsed_s']}s reasoning={rec['reasoning_used']} "
              f"annotations={len(ann)} unavailable_flag={unavailable}")
        print("  --- first 500 chars ---")
        print("  " + content[:500].replace("\n", "\n  "))
        if unavailable:
            failures += 1
    return failures


def run_type(ptype: str) -> None:
    role_task = ROLE_TASKS[ptype]
    base = OUT / ptype
    print(f"\n########## TYPE: {ptype} ##########", flush=True)

    def _stage_call(stage: int, name: str, prompt_text: str) -> tuple[str, dict]:
        rec = call_model(MODELS[name], prompt_text, label=f"{ptype}_s{stage}_{name}")
        save(rec, base / f"stage{stage}_{name}")
        return name, rec

    originals: dict[str, str] = {}
    print("[stage1] launching 3 in parallel ...", flush=True)
    s1_prompt = STAGE1.format(shared=SHARED_CONTEXT, role_task=role_task)
    with ThreadPoolExecutor(max_workers=3) as pool:
        for name, rec in pool.map(lambda n: _stage_call(1, n, s1_prompt), MODELS):
            if rec["ok"] and (p := extract_prompt(rec["content"])):
                originals[name] = p
                print(f"    {name}: ok {rec['elapsed_s']}s, prompt {len(p)} chars")
            else:
                print(f"    {name}: FAILED: {rec.get('error') or 'no prompt block extracted'}")
    if len(originals) < 2:
        print(f"[{ptype}] ABORT: fewer than 2 originals")
        return

    letters = dict(zip(sorted(originals), ["A", "B", "C"]))
    cand_block = "\n\n".join(
        f"## Candidate {letters[n]}\n```\n{originals[n]}\n```" for n in sorted(originals)
    )

    merges: dict[str, str] = {}
    print("[stage2] launching 3 in parallel ...", flush=True)
    s2_prompt = STAGE2.format(shared=SHARED_CONTEXT, role_task=role_task,
                              candidates=cand_block)
    with ThreadPoolExecutor(max_workers=3) as pool:
        for name, rec in pool.map(lambda n: _stage_call(2, n, s2_prompt), MODELS):
            if rec["ok"] and (p := extract_prompt(rec["content"])):
                merges[name] = p
                print(f"    {name}: ok {rec['elapsed_s']}s, prompt {len(p)} chars")
            else:
                print(f"    {name}: FAILED: {rec.get('error') or 'no prompt block extracted'}")

    m_letters = dict(zip(sorted(merges), ["M1", "M2", "M3"]))
    full_block = cand_block + "\n\n" + "\n\n".join(
        f"## Merge {m_letters[n]}\n```\n{merges[n]}\n```" for n in sorted(merges)
    )
    print(f"[stage3] {FINALIZER} ...", flush=True)
    rec = call_model(MODELS[FINALIZER],
                     STAGE3.format(shared=SHARED_CONTEXT, role_task=role_task,
                                   candidates=full_block),
                     label=f"{ptype}_s3_final")
    save(rec, base / "stage3_final")
    if rec["ok"] and (p := extract_prompt(rec["content"])):
        (base / "FINAL_PROMPT.md").write_text(p)
        print(f"    FINAL ok {rec['elapsed_s']}s, prompt {len(p)} chars -> {base}/FINAL_PROMPT.md")
    else:
        print(f"    FINAL FAILED: {rec.get('error') or 'no prompt block extracted'}")


def main() -> None:
    if not KEY:
        sys.exit("no OPENROUTER_API_KEY")
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--type", choices=["panel", "judge", "synthesis"])
    args = ap.parse_args()
    if args.smoke:
        sys.exit(1 if smoke() else 0)
    if args.run:
        for ptype in ([args.type] if args.type else ["panel", "judge", "synthesis"]):
            run_type(ptype)


if __name__ == "__main__":
    main()
