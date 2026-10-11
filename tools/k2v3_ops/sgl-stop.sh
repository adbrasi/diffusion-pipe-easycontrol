#!/bin/bash
pkill -f '^/workspace/env-sgl/bin/python3 /workspace/env-sgl/bin/sglang serve'
for i in $(seq 1 20); do pgrep -f '^/workspace/env-sgl/bin/python3 /workspace/env-sgl/bin/sglang serve' >/dev/null || break; sleep 1; done
pkill -9 -f '^/workspace/env-sgl/bin/python3 /workspace/env-sgl/bin/sglang serve' 2>/dev/null
sleep 2; nvidia-smi --query-gpu=memory.used --format=csv,noheader
