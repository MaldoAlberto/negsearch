#!/usr/bin/env bash
# Publishes this repository to your GitHub account (private by default) and fixes the repository URL in the paper.
# Requires: git, GitHub CLI (`gh auth login` done once).  Usage: ./publish_to_github.sh [repo-name] [--public]
set -euo pipefail
NAME="${1:-negsearch}"; VIS="--private"; [[ "${2:-}" == "--public" ]] && VIS="--public"
USER_LOGIN="$(gh api user --jq .login)"
sed -i "s#github.com/USERNAME/negsearch#github.com/${USER_LOGIN}/${NAME}#" main.tex
git add -A && git commit -q -m "Set repository URL" || true
gh repo create "${USER_LOGIN}/${NAME}" ${VIS} --source=. --remote=origin --push
echo "Published: https://github.com/${USER_LOGIN}/${NAME}"
