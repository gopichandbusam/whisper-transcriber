#!/usr/bin/env bash
set -euo pipefail

APP_MODULE="app.py"
VENV_DIR=".venv"
REQUIREMENTS_FILE="requirements.txt"
WHISPER_MODEL_SIZE="base" # change to small, medium, large, large-v2 as needed
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-5000}"

usage() {
  cat <<EOF
Usage: $0 <command>

Commands:
  setup        Create virtual environment & install dependencies
  run          Run the Flask app (auto setup if needed)
  stop         Stop app started with --background flag
  run-bg       Run the app in background (writes PID file .app.pid)
  status       Show status of background app
  clean        Remove compiled pyc/__pycache__ garbage
  reset        Remove venv & downloaded Whisper model cache (asks confirmation)
  purge        Same as reset plus deletes uploads/ & output/ generated text
  shell        Activate the virtual environment shell

Environment variables you can override:
  WHISPER_MODEL_SIZE (default: base)
  HOST (default: 127.0.0.1)
  PORT (default: 5000)

Examples:
  ./manage.sh setup
  ./manage.sh run
  WHISPER_MODEL_SIZE=small ./manage.sh run
  ./manage.sh run-bg && ./manage.sh status
  ./manage.sh stop
  ./manage.sh purge
EOF
}

command_exists() { command -v "$1" >/dev/null 2>&1; }

ensure_python() {
  if command_exists python3; then
    PYTHON_BIN=python3
  elif command_exists python; then
    PYTHON_BIN=python
  else
    echo "Python 3 not found." >&2
    exit 1
  fi
}

create_venv() {
  ensure_python
  if [ ! -d "$VENV_DIR" ]; then
    echo "[+] Creating virtual environment in $VENV_DIR"
    "$PYTHON_BIN" -m venv "$VENV_DIR"
  fi
}

activate_venv() {
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
  PYTHON_BIN=python
}

install_requirements() {
  activate_venv
  echo "[+] Upgrading pip"
  python -m pip install --upgrade pip
  echo "[+] Installing requirements"
  pip install -r "$REQUIREMENTS_FILE"
  echo "[+] (Optional) Downloading Whisper model: $WHISPER_MODEL_SIZE"
  python - <<PYEOF
import whisper, os
print('Downloading/Loading model:', os.environ.get('WHISPER_MODEL_SIZE','base'))
whisper.load_model(os.environ.get('WHISPER_MODEL_SIZE','base'))
PYEOF
}

setup() {
  create_venv
  install_requirements
  echo "[+] Setup complete"
}

run_app() {
  activate_venv
  export FLASK_ENV=production
  export WHISPER_MODEL_SIZE
  HOST="${HOST:-127.0.0.1}"
  PORT="${PORT:-5000}"
  echo "[+] Starting app on http://$HOST:$PORT"
  export HOST PORT
  exec python "$APP_MODULE"
}

run_app_bg() {
  activate_venv
  export FLASK_ENV=production
  export WHISPER_MODEL_SIZE
  HOST="${HOST:-127.0.0.1}"
  PORT="${PORT:-5000}"
  echo "[+] Starting app in background on http://$HOST:$PORT"
  export HOST PORT
  nohup python "$APP_MODULE" >/tmp/whisper_transcriber.log 2>&1 &
  echo $! > .app.pid
  echo "[+] PID $(cat .app.pid). Logs: /tmp/whisper_transcriber.log"
}

stop_app() {
  if [ -f .app.pid ]; then
    PID=$(cat .app.pid)
    if kill -0 "$PID" 2>/dev/null; then
      echo "[+] Stopping PID $PID"
      kill "$PID" || true
      sleep 1
      if kill -0 "$PID" 2>/dev/null; then
        echo "[!] Force killing $PID"
        kill -9 "$PID" || true
      fi
    else
      echo "[!] Process $PID not running"
    fi
    rm -f .app.pid
  else
    echo "[!] No .app.pid file"
  fi
}

status_app() {
  if [ -f .app.pid ]; then
    PID=$(cat .app.pid)
    if kill -0 "$PID" 2>/dev/null; then
      echo "[+] App running with PID $PID"
    else
      echo "[!] Stale PID file (.app.pid)"
    fi
  else
    echo "[i] App not running (no PID file)"
  fi
}

clean() {
  echo "[+] Removing __pycache__ and *.pyc"
  find . -type d -name __pycache__ -prune -exec rm -rf {} +
  find . -type f -name '*.py[co]' -delete
}

reset() {
  read -r -p "Remove virtual env and model cache (~/.cache/whisper)? [y/N] " ans
  case $ans in
    [yY]*) ;;
    *) echo "Aborted"; return ;;
  esac
  stop_app || true
  rm -rf "$VENV_DIR"
  rm -rf "$HOME/.cache/whisper"
  echo "[+] Reset complete"
}

purge() {
  read -r -p "Remove venv, model cache, uploads/, output/? [y/N] " ans
  case $ans in
    [yY]*) ;;
    *) echo "Aborted"; return ;;
  esac
  stop_app || true
  rm -rf "$VENV_DIR" "$HOME/.cache/whisper" uploads/ output/
  echo "[+] Purge complete"
}

shell_venv() {
  activate_venv
  echo "[+] Virtual environment activated (type 'deactivate' to exit)"
  $SHELL
}

case ${1:-} in
  setup) setup ;;
  run) [ -d "$VENV_DIR" ] || setup; run_app ;;
  run-bg) [ -d "$VENV_DIR" ] || setup; run_app_bg ;;
  stop) stop_app ;;
  status) status_app ;;
  clean) clean ;;
  reset) reset ;;
  purge) purge ;;
  shell) [ -d "$VENV_DIR" ] || setup; shell_venv ;;
  -h|--help|help) usage ;;
  *) usage; exit 1 ;;
esac
