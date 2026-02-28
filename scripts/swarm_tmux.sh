#!/usr/bin/env bash
set -euo pipefail

SESSION="gpunity-swarm"
ROOT="$(pwd)"

# Adjust these to your Claude Code CLI invocation.
# If your CLI uses a different command, replace "claude".
LEADER_MODEL="${LEADER_MODEL:-opus}"
WORKER_MODEL="${WORKER_MODEL:-sonnet}"

tmux new-session -d -s "$SESSION" -c "$ROOT"

# Leader
tmux rename-window -t "$SESSION:0" "leader-opus"
tmux send-keys -t "$SESSION:leader-opus" \
  "claude --model \"$LEADER_MODEL\" --project \"$ROOT\"" C-m

# Workers (7)
declare -a WINS=(
  "profiler"
  "sandbox"
  "agent"
  "validator"
  "reporter"
  "cli-config"
  "test-infra"
)

for w in "${WINS[@]}"; do
  tmux new-window -t "$SESSION" -n "$w" -c "$ROOT"
  tmux send-keys -t "$SESSION:$w" \
    "claude --model \"$WORKER_MODEL\" --project \"$ROOT\"" C-m
done

tmux select-window -t "$SESSION:leader-opus"
tmux attach -t "$SESSION"
