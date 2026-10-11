#!/usr/bin/env bash
# 运行 Dora 冒烟测试。优先用已激活的 venv；否则回退到仓库根的 .venv。
#   ./tools/bench/dora_smoke/run_smoke.sh [--log-level info]
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -z "${VIRTUAL_ENV:-}" ]; then
  REPO="$(cd "$HERE/../../.." && pwd)"
  if [ -x "$REPO/.venv/bin/dora" ]; then
    export VIRTUAL_ENV="$REPO/.venv"
    export PATH="$VIRTUAL_ENV/bin:$PATH"
  fi
fi
exec dora run "$HERE/smoke.yml" "$@"
