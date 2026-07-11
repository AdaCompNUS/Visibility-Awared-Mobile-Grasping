#!/usr/bin/env bash
# Simulate the browser locally by POSTing to the robomesh Flask bridge (ros_interface.py
# on :11111), so you can test the ManiSkill node without the Go server / RoboMesh webapp.
#
#   bash experiments/robomesh/simulate.sh chat  "top down"        # -> /user_instruction
#   bash experiments/robomesh/simulate.sh point 0.5 0.5           # -> /user_point (x,y in [0,1])
set -euo pipefail
HOST="${ROBOMESH_HOST:-127.0.0.1}"
PORT="${ROBOMESH_PORT:-11111}"
KIND="${1:-}"

case "$KIND" in
  chat)
    TEXT="${2:?usage: simulate.sh chat \"<text>\"}"
    curl -sS -X POST "http://${HOST}:${PORT}/chat" \
         -H 'Content-Type: application/json' -d "{\"text\": \"${TEXT}\"}"; echo ;;
  point)
    X="${2:?usage: simulate.sh point <x 0..1> <y 0..1>}"; Y="${3:?need y}"
    curl -sS -X POST "http://${HOST}:${PORT}/point" \
         -H 'Content-Type: application/json' -d "{\"x\": ${X}, \"y\": ${Y}}"; echo ;;
  *)
    echo "usage: $0 chat \"<text>\"   |   $0 point <x> <y>" >&2; exit 1 ;;
esac
