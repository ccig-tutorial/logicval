# Annotation-only Cloud Run image -- no torch/transformers (requirements.txt already
# excludes them by default), so "Run perception on all images" is unavailable once
# deployed; annotators browse already-computed automated results and hand-annotate.
# No ASP/clingo either -- this pipeline never runs the constraint check (see
# src/common/dataset_gen.py), so no clingo binary and no eval_dataset_gen/ solver code
# needed in this image.
#
# Built from a staged context (see deploy.sh), not this directory directly -- the app
# has a hard runtime dependency on the *sibling* ccig-dataset-gen/src/common/ directory
# (src/common/dataset_gen.py imports domain_clevr/domain_coco -- the color/shape/
# material vocabulary -- from it, nothing else), which lives outside a plain
# `docker build .` here's reach. deploy.sh assembles requirements.txt, src/, this
# Dockerfile, and a copy of just ccig-dataset-gen/src/common/ into
# ccig-human-evaluation/.deploy-context/ and builds from there.
#
# WORKDIR matters: app.py derives its data root as two directories up from itself,
# then "/data" (_ROOT.parent / "data", and the same logic in _find_data_root). With
# WORKDIR /srv/ccig-human-evaluation, that resolves to /srv/data -- deploy.sh mounts
# the Cloud Storage bucket at exactly that path, so don't change this without also
# updating the --add-volume-mount path there.
FROM python:3.11-slim

WORKDIR /srv/ccig-human-evaluation

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt gunicorn

COPY src ./src

# Lands at the sibling path common/dataset_gen.py expects at runtime: two directories
# up from WORKDIR, i.e. /srv/ccig-dataset-gen/src/common -- see header comment above.
COPY ccig-dataset-gen-common /srv/ccig-dataset-gen/src/common

ENV PYTHONUNBUFFERED=1
CMD exec gunicorn --bind 0.0.0.0:${PORT:-8080} --workers 1 --threads 8 --timeout 0 src.app:app
