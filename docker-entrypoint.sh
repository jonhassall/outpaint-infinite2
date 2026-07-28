#!/bin/sh
set -eu

mkdir -p /models/huggingface /data/outputs
chown appuser:appuser /models/huggingface /data /data/outputs 2>/dev/null || true

exec gosu appuser "$@"
