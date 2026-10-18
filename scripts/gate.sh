#!/usr/bin/env bash
# Tiered verification gate. See ".claude/CLAUDE.md" -> "Test Cost Discipline".
#
#   scripts/gate.sh targeted <pytest-args...>   ~6s    mutations, refactors, iteration
#   scripts/gate.sh batch [--flat]              ~3min  end of a batch of fixes
#   scripts/gate.sh release                     ~11min ONCE, before commits are presented
#
# Run the cheapest tier that can answer the question being asked.
#
# Collection cost: the farm-runner image exports PYTHONDONTWRITEBYTECODE, and that
# silences pytest's assertion-rewrite cache, so every run re-parses, re-writes and
# re-compiles every test module it collects (measured here: 18.75 s to rewrite 278
# modules against 0.15 s to load their cache). Every pytest run below therefore goes
# through run_pytest, which drops the variable from the environment it hands on.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 2
# shellcheck disable=SC1091
source .venv/bin/activate || { echo "gate: .venv missing"; exit 2; }
export PYTHONPATH=.

BUNDLES=(open_webui_openrouter_pipe_bundled.py
         open_webui_openrouter_pipe_bundled_compressed.py
         open_webui_openrouter_pipe_bundled_no_plugins.py
         open_webui_openrouter_pipe_bundled_compressed_no_plugins.py)
rc=0
ARGV=("$@")
MODE="${1:-}"
CALL_LIMIT_SECONDS=2.0

# A bundled suite costs ~7 minutes each and answers a question that only matters once the work is
# finished. Asking for one mid-work is almost always a reflex, so it has to be said out loud.
require_deliberate_bundle_run() {
  local what="$1" cost="$2"
  for a in "$@" "${ARGV[@]}"; do
    if [ "$a" = "--yes-i-need-to-waste-the-time-now=1" ]; then
      export GATE_BUNDLE_RUN_APPROVED=1
      return 0
    fi
  done
  cat >&2 <<BANNER

  ========================================================================
  STOP. You asked for $what, which costs about $cost.

  Are you sure you need this MID-WORK? Bundled suites answer one question:
  does the code still behave once it has been flattened into a single file.
  That question cannot change until the work is finished, so the answer you
  get now is an answer you will have to get again at the end anyway.

  The package suite is what tells you whether your change is correct, and
  "scripts/gate.sh batch" runs it without any of this.

  If you genuinely need it now, say so:
      scripts/gate.sh $* --yes-i-need-to-waste-the-time-now=1
  ========================================================================

BANNER
  exit 3
}

step() { printf '\n== %s\n' "$1"; }
keep() { "$@" || rc=1; }

# Run a command with the image's bytecode suppression removed, so pytest can write and
# then read its assertion-rewrite cache (see the header). It is an `env` passthrough
# rather than a pytest wrapper so the two bundled call sites can still set
# OWUI_PIPE_BUNDLE_PATH in the same breath: run_pytest OWUI_PIPE_BUNDLE_PATH="$b" \
# python -m pytest ...
run_pytest() { env -u PYTHONDONTWRITEBYTECODE "$@"; }

build_bundles() {
  step "build bundles (all four)"
  keep python scripts/bundle_v2.py                          >/dev/null
  keep python scripts/bundle_v2.py --compress               >/dev/null
  keep python scripts/bundle_v2.py --no-plugins             >/dev/null
  keep python scripts/bundle_v2.py --compress --no-plugins  >/dev/null
  local newest stale=0
  newest=$(stat -c %Y open_webui_openrouter_pipe/pipe.py)
  for b in "${BUNDLES[@]}"; do
    [ -f "$b" ] && [ "$(stat -c %Y "$b")" -ge "$newest" ] || { echo "gate: STALE OR MISSING $b"; stale=1; }
  done
  [ $stale -eq 0 ] || rc=1
}

package_suite() {
  step "package suite"
  local log
  log=$(mktemp)
  run_pytest python -m pytest tests/ -q --durations=0 --durations-min=1.0 2>&1 | tee "$log"
  if [ "${PIPESTATUS[0]}" -ne 0 ]; then
    rc=1
    echo "call limit not checked: the package suite failed"
  else
    step "call limit (${CALL_LIMIT_SECONDS}s per test, re-timed alone before it fails)"
    keep python scripts/check_slow_tests.py "$log" --limit "$CALL_LIMIT_SECONDS"
  fi
  rm -f "$log"
}

# The batch tier is otherwise blind to a test module that cannot IMPORT under a smaller
# artifact: its rationale is that bundle-only defects are static, which is true of defects
# in the bundle and false of a collection error. One unguarded plugin import aborted both
# --no-plugins modes at collection, so they ran ZERO tests and exited 2, in the release
# tier and in CI. Seconds, not minutes. The step reports its own failure: on a non-zero
# exit it names the artifact, every failing module and at least one exception line.
collect_no_plugins() {
  step "collect-only under the two --no-plugins artifacts"
  local b out err
  for b in open_webui_openrouter_pipe_bundled_no_plugins.py \
           open_webui_openrouter_pipe_bundled_compressed_no_plugins.py; do
    out=$(mktemp); err=$(mktemp)
    if run_pytest OWUI_PIPE_BUNDLE_PATH="$b" python -m pytest tests/ -q --collect-only >"$out" 2>"$err"; then
      # stderr was never redirected before, so the success path must neither swallow it nor
      # re-emit the collected node-id list pytest printed to stdout:
      cat "$err" >&2
      rm -f "$out" "$err"
    else
      rc=1
      echo "  collect-only FAILED under $b; last 20 lines:"
      cat "$out" "$err" | tail -20
      cat "$out" "$err" | grep -E '^ERROR tests/'   # every failing module, unbounded
      cat "$out" "$err" | grep -E '^E   ' | head -20
      rm -f "$out" "$err"
    fi
  done
}

# Ruff's EXE001 (a file that starts with "#!" but is not executable) is not enforced on WSL,
# where this gate runs, and git here records a new file as 100644 (core.fileMode=false). CI
# checks out the index mode and enforces the rule, so the mode is read from the index.
shebang_modes() {
  python - <<'PY'
import subprocess
listing = subprocess.run(["git", "ls-files", "-s", "-z"], capture_output=True, check=True).stdout
bad = []
for entry in filter(None, listing.split(b"\0")):
    meta, path = entry.split(b"\t", 1)
    if not meta.startswith(b"100644 "):
        continue
    try:
        with open(path, "rb") as fh:
            starts = fh.read(2)
    except OSError:
        continue
    if starts == b"#!":
        bad.append(path.decode(errors="replace"))
for p in bad:
    print(f"  {p} starts with #! but git stores it as 100644; fix: git update-index --chmod=+x -- {p}")
print(f"{len(bad)} tracked file(s) start with #! without the executable mode")
raise SystemExit(1 if bad else 0)
PY
}

static_checks() {
  step "ruff (source)";  keep ruff check open_webui_openrouter_pipe/ scripts/ filters/
  step "executable mode of files that start with #! (ruff EXE001, which WSL does not enforce)"
  keep shebang_modes
  step "ruff (bundles)"; keep ruff check --per-file-ignores '*:E402' "${BUNDLES[@]}"
  step "pyright";        keep pyright
  step "prose gate";     keep python scripts/check_added_prose.py HEAD --path open_webui_openrouter_pipe
  step "prescription ledger"
  if [ -f .git/panel-check.py ]; then
    keep python .git/panel-check.py ${STRICT_PRESCRIPTIONS:+--strict}
  else
    echo "skipped: .git/panel-check.py is a local tool and is not present here"
  fi
  step "fragile separators"
  keep python -c "
s=open('open_webui_openrouter_pipe/core/utils.py',encoding='utf-8').read()
a,b=s.count(chr(0x2028)),s.count(chr(0x2029))
print(f'U+2028={a} U+2029={b}')
raise SystemExit(0 if a==1 and b==1 else 1)"
}

case "${1:-}" in
  targeted)
    shift
    [ $# -gt 0 ] || { echo "gate targeted: give pytest node ids or -k expression"; exit 2; }
    step "targeted run -- proves ONLY the nodes named below"
    run_pytest python -m pytest "$@" -q
    rc=$?
    echo
    echo "NOTE: a targeted run is evidence about these nodes ONLY."
    echo "      It is the correct tier for a mutation or refactor."
    echo "      It is NOT grounds to claim 'tests pass' -- that needs: gate.sh batch"
    ;;
  batch)
    [ "${2:-}" = "--flat" ] && require_deliberate_bundle_run "the flat bundled suite" "7 minutes" "$@"
    build_bundles
    package_suite
    static_checks
    collect_no_plugins
    if [ "${2:-}" = "--flat" ]; then
      step "flat bundled suite (imports / module structure / test doubles changed)"
      keep run_pytest OWUI_PIPE_BUNDLE_PATH=open_webui_openrouter_pipe_bundled.py python -m pytest tests/ -q
    fi
    echo
    echo "NOTE: batch tier does NOT run the four bundled suites."
    echo "      Bundle-only defects (from-import attribute access, module-global collisions,"
    echo "      loop-variable shadowing) are STATIC and are covered above by ruff+pyright,"
    echo "      and collection under the two --no-plugins artifacts is checked directly."
    echo "      Runtime bundle behaviour is covered only by: gate.sh batch --flat, or gate.sh release"
    ;;
  release)
    require_deliberate_bundle_run "the release tier, which runs all four bundled suites" "25 minutes" "$@"
    export STRICT_PRESCRIPTIONS=1
    build_bundles
    package_suite
    static_checks
    for b in "${BUNDLES[@]}"; do
      step "bundled suite: $b"
      keep run_pytest OWUI_PIPE_BUNDLE_PATH="$b" python -m pytest tests/ -q
    done
    step "compileall"; keep python -m compileall -q open_webui_openrouter_pipe >/dev/null
    ;;
  *)
    sed -n '2,8p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    exit 2
    ;;
esac

echo
if [ $rc -eq 0 ]; then echo "gate: PASS (${MODE})"; else echo "gate: FAIL (${MODE})"; fi
exit $rc
