# Deploying to Google Cloud Run

`deploy.sh` builds a self-contained image (annotation only — no torch/
transformers, no ASP/clingo) and deploys it to Cloud Run, with your `data/`
folder uploaded once to a Cloud Storage bucket and mounted into the container
so reads/writes work the same as local files. See the main [README](README.md)
for what the app itself does; this file is just the deployment path.

## Prerequisites

- A GCP project that exists and has billing enabled.
- `gcloud auth login` with an account that has access to it.
- `data/` present somewhere reachable — either locally where you run
  `deploy.sh`, or already uploaded to the target bucket (see below).

## Running it

```bash
gcloud auth login   # once, if not already
CCIG_HUMAN_EVAL_PASSWORD='choose-a-real-password' ./deploy.sh
```

Optional overrides (export before running, or prefix the command with them):
`PROJECT` (default `ccig`), `REGION` (default `us-east4`), `BUCKET` (default
`<PROJECT>-human-eval-data`), `SERVICE_NAME` (default `ccig-human-eval`). See
the comments at the top of `deploy.sh` for details.

It prints a public `https://*.run.app` URL when done — share that and the
password with your annotators (see the main README's Password protection
section for how that gate works).

## Managing the service

```bash
# block all access (keeps everything else -- config, revision, data)
gcloud run services update ccig-human-eval --project=ccig-498516 --region=us-east4 --no-allow-unauthenticated

# re-allow public access
gcloud run services update ccig-human-eval --project=ccig-498516 --region=us-east4 --allow-unauthenticated

# delete the service entirely (bucket + image untouched, redeploy with ./deploy.sh)
gcloud run services delete ccig-human-eval --project=ccig-498516 --region=us-east4

# check status
gcloud run services describe ccig-human-eval --project=ccig-498516 --region=us-east4
```

Cost note: the service already scales to zero when idle (no `--min-instances`
set), so it costs nothing while unused either way -- blocking access above is
for reachability, not billing. The only ongoing cost is the bucket's storage
(~$0.02/GB/month), which persists regardless of the service's state.

## Data layout the app expects in the bucket

The whole bucket is mounted at `/srv/data` inside the container, so the
bucket's **root** must directly mirror what's inside your local `data/`
folder — no extra wrapping folder:

```
gs://<bucket>/generated_images/<model>/<dataset>/<id>-<field>.png
gs://<bucket>/evaluation/<model>/<dataset>_auto-perception.json
gs://<bucket>/evaluation/<model>/<dataset>_human-perception.json
gs://<bucket>/ccig_eval_dataset/<dataset>.jsonl
```

If you upload manually through the Cloud Storage console instead of letting
`deploy.sh`'s `gsutil rsync` do it, double-check the result actually looks
like this — it's easy to end up one level too deep (e.g. a literal `data/`
folder sitting at the bucket root) depending on how you select what to drag in.

## Troubleshooting (real issues hit getting this working)

**`gcloud` says `invalid_grant` / asks you to re-login.** Cached credentials
expired. Run `gcloud auth login` (needs an interactive browser flow — can't be
done from a non-interactive script or CI).

**"Project 'X' not found or permission denied" even though the project
exists.** The project's *display name* and its *project ID* are usually
different — IDs are globally unique across all of GCP, so a short name like
`ccig` is often already taken by someone else, and your actual project gets
assigned an ID like `ccig-498516`. Find yours with:
```bash
gcloud projects list --filter="name:<your-display-name>"
```
and pass the real `PROJECT_ID` (not the display name) as `PROJECT=`.

**Cloning your GitHub repo fails with `SAML SSO` errors.** If the repo's org
enforces SAML SSO, an SSH key or personal access token being on your GitHub
account isn't enough — it also has to be individually authorized for that
org: GitHub → Settings → SSH and GPG keys (or Developer settings → Personal
access tokens) → find the key/token → **Configure SSO** → authorize it.

**`gsutil rsync` fails with "does not name a directory".** `data/` isn't
present where you're running `deploy.sh` from (e.g. a fresh clone that never
had 5.5GB of data pushed to it — `data/` is not meant to live in git).
`deploy.sh` skips the upload gracefully in this case and assumes it's already
in the bucket from elsewhere; if it isn't yet, upload it separately (from
wherever `data/` actually lives) before or after deploying.

**Container crashes on startup with `ImportError: cannot import name
'domain_coco' from 'common' (unknown location)`.** The bundled copy of
`ccig-dataset-gen/src/common/` is incomplete — missing `domain_coco.py`
and/or `__init__.py`. This happens if you're deploying from a manually
curated subset of the repo rather than a full checkout; make sure
`ccig-dataset-gen/src/common/` came along in full.

**Deployed, URL returns 503.** Check `gcloud run services describe` for the
service's `status.conditions` first (a `Ready: True` with everything else
green means Cloud Run itself thinks it's healthy — the problem is inside the
container, not the deploy). Then pull the actual startup logs:
```bash
gcloud logging read 'resource.type=cloud_run_revision AND resource.labels.service_name=<service>' \
  --project=<project> --limit=50 --freshness=1d
```
This is what caught both the `domain_coco` import error and would catch most
other startup-time crashes.
