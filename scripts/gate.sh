#!/usr/bin/env bash
# Tiered verification gate. See ".claude/CLAUDE.md" -> "Test Cost Discipline".
#
#   scripts/gate.sh targeted <pytest-args...>   ~6s    mutations, refactors, iteration
#   scripts/gate.sh batch [--flat]              ~3min  end of a batch of fixes
#   scripts/gate.sh release                     ~11min ONCE, before commits are presented
#
# Run the cheapest tier that can answer the question being asked.
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
    if env OWUI_PIPE_BUNDLE_PATH="$b" python -m pytest tests/ -q --collect-only >"$out" 2>"$err"; then
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

static_checks() {
  step "ruff (source)";  keep ruff check open_webui_openrouter_pipe/ scripts/ filters/
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
    python -m pytest "$@" -q
    rc=$?
    echo
    echo "NOTE: a targeted run is evidence about these nodes ONLY."
    echo "      It is the correct tier for a mutation or refactor."
    echo "      It is NOT grounds to claim 'tests pass' -- that needs: gate.sh batch"
    ;;
  batch)
    [ "${2:-}" = "--flat" ] && require_deliberate_bundle_run "the flat bundled suite" "7 minutes" "$@"
    build_bundles
    step "package suite"; keep python -m pytest tests/ -q
    static_checks
    collect_no_plugins
    if [ "${2:-}" = "--flat" ]; then
      step "flat bundled suite (imports / module structure / test doubles changed)"
      keep env OWUI_PIPE_BUNDLE_PATH=open_webui_openrouter_pipe_bundled.py python -m pytest tests/ -q
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
    step "package suite"; keep python -m pytest tests/ -q
    static_checks
    for b in "${BUNDLES[@]}"; do
      step "bundled suite: $b"
      keep env OWUI_PIPE_BUNDLE_PATH="$b" python -m pytest tests/ -q
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
