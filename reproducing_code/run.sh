#!/bin/sh
set -eu
cd "$(dirname "$0")"

if [ -x .venv/bin/python ]; then
    PYTHON=.venv/bin/python
else
    PYTHON=python3
fi

MODE="${1:-cached}"
if [ "$#" -gt 0 ]; then
    shift
fi
case "$MODE" in
    cached)
        exec "$PYTHON" reproduce.py --mode cached "$@"
        ;;
    regenerate)
        exec "$PYTHON" regenerate.py "$@"
        ;;
    *)
        echo "Usage: $0 [cached|regenerate]" >&2
        exit 2
        ;;
esac
