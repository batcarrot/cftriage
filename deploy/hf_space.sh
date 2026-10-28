#!/usr/bin/env bash
# Publish the hosted web app to a Hugging Face Space (Docker SDK).
#
#   1. Create a Space at https://huggingface.co/new-space (SDK: Docker, Blank).
#   2. Log in once:  hf auth login   (a token with write access)
#   3. Run:          deploy/hf_space.sh <your-hf-username>/<space-name>
#
# The Space gets the committed files from HEAD plus a README with the Space
# config. Uncommitted changes are not deployed.
set -euo pipefail
space="${1:?usage: deploy/hf_space.sh <user>/<space>}"
root="$(git rev-parse --show-toplevel)"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

git -C "$root" archive HEAD | tar -x -C "$tmp"
rm -rf "$tmp/demo" "$tmp/tests"
{
  cat <<'YAML'
---
title: cftriage
emoji: 🩺
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Does an AI triage women and men differently?
---

YAML
  cat "$root/README.md"
} > "$tmp/README.md"

token="$(python3 -c 'from huggingface_hub import get_token; print(get_token() or "")' 2>/dev/null || true)"
[ -n "$token" ] || { echo "Not logged in to Hugging Face. Run: hf auth login" >&2; exit 1; }
user="$(python3 -c 'from huggingface_hub import whoami; print(whoami()["name"])')"

cd "$tmp"
git init -q -b main
git add -A
git -c user.name="$user" -c user.email="$user@users.noreply.huggingface.co" \
  commit -q -m "Deploy cftriage $(git -C "$root" rev-parse --short HEAD)"
git push -q --force "https://$user:$token@huggingface.co/spaces/$space" main
echo "Deployed. Building at https://huggingface.co/spaces/$space"
