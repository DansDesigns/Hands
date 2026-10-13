#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "$0")"
exec ./run_linux.sh "$@"
