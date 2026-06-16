#!/usr/bin/env bash
set -euo pipefail

VERSION="${1:-}"
MSG="${2:-}"

if [ -z "$VERSION" ]; then
  echo "Usage: bash commands/tag_version.sh v0.2.2 \"v0.2.2 short message\""
  exit 1
fi
if [ -z "$MSG" ]; then
  MSG="$VERSION"
fi

if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "ERROR: working tree has uncommitted changes. Commit or reset before tagging."
  git status --short
  exit 2
fi

git fetch --tags origin
if git rev-parse "$VERSION" >/dev/null 2>&1; then
  echo "ERROR: tag already exists: $VERSION"
  exit 3
fi

git tag -a "$VERSION" -m "$MSG"
git push origin "$VERSION"

echo "Created and pushed tag: $VERSION"
