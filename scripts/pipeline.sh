#!/usr/bin/env bash
set -euo pipefail

# Canonical operator entrypoint for repo pipelines.
#
# Usage:
#   scripts/pipeline.sh launch so101-eval
#   scripts/pipeline.sh check so101-eval
#   scripts/pipeline.sh stop so101-eval
#   scripts/pipeline.sh restart so101-eval
#   scripts/pipeline.sh status so101-eval

REPO_ROOT="${BLUPE_EVALS_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

usage() {
  cat <<'EOF'
Usage:
  scripts/pipeline.sh <command> <pipeline>

Commands:
  launch    Start the pipeline and wait for readiness.
  check     Run the pipeline readiness check.
  status    Alias for check.
  stop      Stop the pipeline.
  restart   Stop, then launch the pipeline.

Pipelines:
  so101-eval    SO101 + MolmoAct2 + SAM eval stack.

Examples:
  scripts/pipeline.sh launch so101-eval
  scripts/pipeline.sh check so101-eval
  scripts/pipeline.sh stop so101-eval
EOF
}

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
  usage
  exit 0
fi

COMMAND="${1:-}"
PIPELINE="${2:-}"

if [ -z "$COMMAND" ] || [ -z "$PIPELINE" ]; then
  usage >&2
  exit 2
fi

case "$PIPELINE" in
  so101-eval|so101)
    case "$COMMAND" in
      launch|start)
        exec "$REPO_ROOT/scripts/start_so101_eval_stack.sh"
        ;;
      check|status)
        exec "$REPO_ROOT/scripts/check_so101_eval_stack.sh"
        ;;
      stop)
        exec "$REPO_ROOT/scripts/stop_so101_eval_stack.sh"
        ;;
      restart)
        "$REPO_ROOT/scripts/stop_so101_eval_stack.sh"
        exec "$REPO_ROOT/scripts/start_so101_eval_stack.sh"
        ;;
      *)
        echo "Unknown command for so101-eval: ${COMMAND}" >&2
        usage >&2
        exit 2
        ;;
    esac
    ;;
  *)
    echo "Unknown pipeline: ${PIPELINE}" >&2
    usage >&2
    exit 2
    ;;
esac
