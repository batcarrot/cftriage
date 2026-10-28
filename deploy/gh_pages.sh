#!/usr/bin/env bash
# Publish the static site to GitHub Pages (the gh-pages branch of origin).
#
#   deploy/gh_pages.sh
#
# Builds from the working tree with static/build.py and force-pushes the
# result. Commit first so the site matches main.
set -euo pipefail
root="$(git rev-parse --show-toplevel)"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

python3 "$root/static/build.py" "$tmp/site"
cd "$tmp/site"
git init -q -b gh-pages
git add -A
git commit -q -m "Deploy $(git -C "$root" rev-parse --short HEAD)"
git push -q --force "$(git -C "$root" remote get-url origin)" gh-pages
echo "Pushed to gh-pages. GitHub Pages updates within a minute or two."
