# syntax=docker/dockerfile:1.7
ARG CUDA_TAG=13.0.1

FROM nvidia/cuda:${CUDA_TAG}-devel-ubuntu24.04 AS base

ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
ENV TORCH_CUDA_ARCH_LIST="8.9;10.0;12.0"

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get -o Acquire::ForceIPv4=true update && \
    apt-get -o Acquire::ForceIPv4=true install -y --no-install-recommends \
      python3 \
      python3-venv \
      python3-dev \
      python3-pip \
      build-essential \
      ninja-build \
      ca-certificates \
      git \
      dcmtk \
      libgdcm-tools \
      dcm2niix \
      dicom3tools \
      libjpeg-turbo8 \
      libopenjp2-7 \
      libtiff6 \
      zlib1g

RUN python3 -m venv /opt/venv
ENV VIRTUAL_ENV=/opt/venv
ENV PATH=/opt/venv/bin:${PATH}
ENV PIP_NO_CACHE_DIR=1

# Upgrade pip and build tooling
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    python -m pip install --upgrade pip setuptools wheel ninja

RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    pip install --no-cache-dir \
      torch==2.9.0 \
      torchvision==0.24.0 \
      torchaudio==2.9.0 \
      --index-url https://download.pytorch.org/whl/cu130

RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    pip install --no-cache-dir \
      pydicom \
      pynetdicom \
      dicomweb-client \
      highdicom \
      pylibjpeg \
      pylibjpeg-libjpeg \
      pylibjpeg-openjpeg

# Build context (pruned via .dockerignore)
WORKDIR /build
COPY . .

# Install the main package into the venv
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    pip install --no-cache-dir .

# Build and install fused ops into the same venv
RUN cd fused_ops && \
    python setup.py build_ext && \
    python setup.py install

# --------------------------
# Minimal runtime image
# --------------------------
FROM nvidia/cuda:${CUDA_TAG}-runtime-ubuntu24.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Only runtime necessities
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get -o Acquire::ForceIPv4=true update && \
    apt-get -o Acquire::ForceIPv4=true install -y --no-install-recommends \
      python3 \
      ca-certificates \
      git \
      dcmtk \
      libgdcm-tools \
      dcm2niix \
      dicom3tools \
      libjpeg-turbo8 \
      libopenjp2-7 \
      libtiff6 \
      zlib1g

# Bring the prebuilt virtualenv from the base stage
COPY --from=base /opt/venv /opt/venv

# Use the venv by default
ENV PATH=/opt/venv/bin:${PATH}
ENV VIRTUAL_ENV=/opt/venv

WORKDIR /app

# Default: print versions (override with your own CMD/ENTRYPOINT)
ENTRYPOINT ["/bin/bash"]
