#!/usr/bin/env bash
# Serve the reviewer model on a Lambda GPU instance and tunnel it to localhost.
#
#   ./lambda_reviewer.sh setup     install Ollama on the instance and pull the model
#   ./lambda_reviewer.sh tunnel    forward localhost:11434 -> instance Ollama (stays open)
#   ./lambda_reviewer.sh status    show GPU and loaded models on the instance
#
# Reads LAMBDA_HOST, LAMBDA_SSH_KEY and optional LAMBDA_MODEL from .env.local.
set -euo pipefail
cd "$(dirname "$0")"
set -a; [ -f .env.local ] && . ./.env.local; set +a
: "${LAMBDA_HOST:?set LAMBDA_HOST in .env.local}"
KEY="${LAMBDA_SSH_KEY:-~/.ssh/lambda.pem}"; KEY="${KEY/#\~/$HOME}"
MODEL="${LAMBDA_MODEL:-gemma3:12b}"
SSH="ssh -i $KEY -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30 ubuntu@$LAMBDA_HOST"

case "${1:-}" in
  setup)
    $SSH bash -s "$MODEL" <<'EOF'
set -e
MODEL="$1"
if ! command -v ollama >/dev/null; then
  curl -fsSL https://ollama.com/install.sh | sh
fi
sudo systemctl enable --now ollama 2>/dev/null || (nohup ollama serve >/tmp/ollama.log 2>&1 &)
sleep 3
ollama pull "$MODEL"
echo "--- models on $(hostname) ---"; ollama list
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader
EOF
    ;;
  tunnel)
    echo "forwarding localhost:11434 -> $LAMBDA_HOST:11434 (ctrl-c to stop)"
    exec $SSH -N -L 11434:127.0.0.1:11434
    ;;
  status)
    $SSH 'nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader; ollama list; ollama ps'
    ;;
  *)
    sed -n 2,9p "$0"; exit 1 ;;
esac
