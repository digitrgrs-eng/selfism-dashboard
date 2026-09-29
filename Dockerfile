# syntax=docker/dockerfile:1.7
FROM python:3.12-slim AS r2deps
RUN pip install --no-cache-dir --no-compile --target=/r2deps boto3==1.43.104
# Verified linux/amd64 manifest of 10sorllabs/comfyui-workflow-launcher:2.0.
# Reuse its already-built CUDA/runtime layers. No RUN may be added here:
# COPY --link allows BuildKit to publish without unpacking the CUDA rootfs.
FROM 10sorllabs/comfyui-workflow-launcher@sha256:d01908958aa33cc9117b478d81845d14ed53c135f173e3db3eafe14e780e8846

ARG IMAGE_VERSION=dev
LABEL org.opencontainers.image.title="Selfora / Selfism dashboard" \
      org.opencontainers.image.description="10sorLabs launcher with Selfism installation and runtime repair" \
      org.opencontainers.image.source="https://github.com/digitrgrs-eng/selfism-dashboard" \
      org.opencontainers.image.version="${IMAGE_VERSION}"

ENV LAUNCHER_AUTO_UPDATE=0 \
    SELFISM_AUTO_REPAIR=1 \
    HF_TOKEN_FILE=/dev/null \
    PYTHONPATH=/opt/r2deps:/opt/10sorlabs

# User credentials must be supplied via RunPod environment variables.
# Do not use credentials possibly baked into the upstream image.
COPY --link launcher/ /opt/10sorlabs/launcher/
COPY --link --from=r2deps /r2deps/ /opt/r2deps/
COPY --link scripts/upload_models_r2.py /opt/10sorlabs/scripts/upload_models_r2.py
COPY --link catalog/ /opt/10sorlabs/catalog/
COPY --link selfism_workflows/ /opt/10sorlabs/selfism_workflows/
COPY --link --chmod=755 docker/entrypoint.sh /start.sh

# Inherit the original services, ports, working directory and entrypoint.
