#!/usr/bin/env bash
set -e

# Render define PORT no ambiente.
# Gunicorn para produção com 2 workers e threads.
gunicorn -w 2 -k gthread -t 120 -b 0.0.0.0:$PORT main:app
