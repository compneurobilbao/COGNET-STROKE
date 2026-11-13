# Contract step image for COGNET-STROKE (card F0-26). Build from the repository root:
#   docker build -f step/Dockerfile -t cognet-stroke-step:dev .
# Only the slim runtime package (runtime/) is installed; the analysis environment of the
# repository (top-level pyproject.toml) is not part of the image. Concept maps and null maps are
# not in the image either: they are the concept-maps bundle mounted at /ref/concept_maps.

# ---- Contract SDK and contract files, pinned to a commit of the public stroke-server repo ----
FROM python:3.12.3-slim-bookworm AS contract
ARG STROKE_SERVER_SHA=f6c8dc57ef9629e53bff2a26bbe66202465625bb
RUN apt-get update -qq && apt-get install -y -q --no-install-recommends ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /src
RUN curl -fsSL "https://codeload.github.com/compneurobilbao/stroke-server-biobizkaia/tar.gz/${STROKE_SERVER_SHA}" \
    | tar -xz --strip-components=1 --wildcards '*/steps/sdk' '*/contracts' \
    && test -f steps/sdk/pyproject.toml && test -f contracts/validate.py \
    && test -f contracts/schemas/v1/cognet-profile.schema.json

# ---- Step image ----
FROM python:3.12.3-slim-bookworm
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/opt/step/src \
    STEP_CONTRACTS_DIR=/opt/step/contracts \
    MPLCONFIGDIR=/tmp/matplotlib \
    PATH="/usr/local/bin:/usr/bin:/bin"

COPY step/requirements-image.txt /opt/build/requirements-image.txt
COPY runtime /opt/build/runtime
COPY --from=contract /src/steps/sdk /opt/build/sdk
RUN pip install -r /opt/build/requirements-image.txt \
    && pip install --no-deps /opt/build/runtime \
    && pip install --no-deps /opt/build/sdk \
    && rm -rf /opt/build

COPY --from=contract /src/contracts/schemas /opt/step/contracts/schemas
COPY --from=contract /src/contracts/artefact-types.yaml /src/contracts/validate.py /opt/step/contracts/
COPY step/src /opt/step/src
COPY step/step.yaml /opt/step/step.yaml
COPY step/bin/run /usr/local/bin/run

RUN mkdir -p /in /out /ref /tmp \
    && chmod 0755 /in /out /ref /usr/local/bin/run \
    && chmod 1777 /tmp \
    && chmod -R a+rX /opt/step \
    && PYTHONDONTWRITEBYTECODE= python -m compileall -q /opt/step/src /opt/step/contracts \
    && chmod -R a+rX /opt/step /usr/local/lib/python3.12/site-packages \
    && MPLBACKEND=Agg python -c "import cognet_runtime, cognet_step, stroke_step_sdk, nilearn.plotting"

USER 10001:10001
WORKDIR /tmp
CMD ["/usr/local/bin/run", "--help"]
