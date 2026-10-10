#!/usr/bin/env bash
# One-shot setup + deploy of ccig-human-evaluation to Cloud Run, on the "ccig" GCP
# project, from scratch. Run from anywhere -- it cd's to the repo root itself.
#
# Prerequisites this script does NOT do for you:
#   - `gcloud auth login` with valid (non-expired) credentials
#   - the "ccig" project must already exist and have billing enabled
#   - CCIG_HUMAN_EVAL_PASSWORD must be set in your shell before running this --
#     never hardcode a real password into this file, it's meant to be committed
#
# Usage:
#   CCIG_HUMAN_EVAL_PASSWORD='choose-a-real-password' ./deploy.sh
#
# Optional overrides (export before running):
#   PROJECT (default: ccig)
#   REGION (default: us-east4)
#   BUCKET (default: <PROJECT>-human-eval-data -- GCS bucket names are globally
#           unique across ALL of GCS, not just your project; override this if
#           that name is already taken by someone else)
#   SERVICE_NAME (default: ccig-human-eval)
#
# Safe to re-run: bucket creation and the data sync are both idempotent, and
# `gcloud run deploy` updates the existing service in place rather than duplicating
# it.

set -euo pipefail

PROJECT="${PROJECT:-ccig}"
REGION="${REGION:-us-east4}"
BUCKET="${BUCKET:-${PROJECT}-human-eval-data}"
SERVICE_NAME="${SERVICE_NAME:-ccig-human-eval}"

if [ -z "${CCIG_HUMAN_EVAL_PASSWORD:-}" ]; then
  echo "ERROR: set CCIG_HUMAN_EVAL_PASSWORD before running this script, e.g.:" >&2
  echo "  CCIG_HUMAN_EVAL_PASSWORD='your-password' $0" >&2
  exit 1
fi

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$APP_DIR/.." && pwd)"
cd "$REPO_ROOT"

echo "==> Project: $PROJECT   Region: $REGION   Bucket: gs://$BUCKET   Service: $SERVICE_NAME"

echo "==> Staging a self-contained build context (app.py needs the sibling"
echo "    ccig-dataset-gen/src/common/ directory -- just the color/shape/material"
echo "    ccig-human-evaluation/ alone can't reach) into ccig-human-evaluation/.deploy-context/ ..."
DEPLOY_CTX="$APP_DIR/.deploy-context"
rm -rf "$DEPLOY_CTX"
mkdir -p "$DEPLOY_CTX"
cp "$APP_DIR/Dockerfile" "$DEPLOY_CTX/"
cp "$APP_DIR/requirements.txt" "$DEPLOY_CTX/"
cp -r "$APP_DIR/src" "$DEPLOY_CTX/src"
cp -r "$REPO_ROOT/ccig-dataset-gen/src/common" "$DEPLOY_CTX/ccig-dataset-gen-common"

echo "==> Enabling required APIs..."
gcloud services enable run.googleapis.com cloudbuild.googleapis.com storage.googleapis.com \
  --project="$PROJECT"

echo "==> Creating bucket (skips if it already exists)..."
gsutil mb -p "$PROJECT" -l "$REGION" "gs://$BUCKET" 2>/dev/null || echo "    (bucket already exists, continuing)"

if [ -d "data" ]; then
  echo "==> Syncing data/ to gs://$BUCKET (uploads all ~5.5GB the first time; only changes after that)..."
  # To the bucket ROOT, not gs://$BUCKET/data -- the Cloud Run deploy below mounts the
  # whole bucket at /srv/data, so the bucket's root must directly mirror what's inside
  # your local data/ folder (gs://$BUCKET/generated_images/..., not
  # gs://$BUCKET/data/generated_images/...), or the app looks for files one level too deep.
  gsutil -m rsync -r data/ "gs://$BUCKET"
else
  echo "==> No local data/ directory here (e.g. running from a fresh clone that never had"
  echo "    data/ pushed to it) -- skipping upload, assuming it's already in the bucket"
  echo "    (uploaded separately, from wherever data/ actually lives)."
fi

echo "==> Deploying to Cloud Run..."
gcloud run deploy "$SERVICE_NAME" \
  --project="$PROJECT" \
  --source="$DEPLOY_CTX" \
  --region="$REGION" \
  --allow-unauthenticated \
  --execution-environment=gen2 \
  --port=8080 \
  --memory=1Gi \
  --cpu=1 \
  --max-instances=1 \
  --set-env-vars="CCIG_HUMAN_EVAL_PASSWORD=${CCIG_HUMAN_EVAL_PASSWORD}" \
  --add-volume=name=data-vol,type=cloud-storage,bucket="$BUCKET",readonly=false \
  --add-volume-mount=volume=data-vol,mount-path=/srv/data

echo
echo "==> Done. In the deployed app's setup form, paths live under /srv/data/, e.g.:"
echo "      Images dir:   /srv/data/generated_images/<model>/<dataset>"
echo "      Prompts file: /srv/data/ccig_eval_dataset/<dataset>.jsonl"
echo "    /browse auto-discovers files anywhere under /srv/data/evaluation/ -- nothing to configure there."
