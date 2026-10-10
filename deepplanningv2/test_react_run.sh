#!/bin/bash
# Run the same test as test_run.sh with shoppingplanning/reactAgent.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export SHOPPING_AGENT_BACKEND=react
exec bash "$ROOT/test_run.sh" "$@"
