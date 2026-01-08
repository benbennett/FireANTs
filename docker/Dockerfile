# syntax=docker/dockerfile:1.7
ARG CUDA_TAG=13.0.1
FROM nvidia/cuda:${CUDA_TAG}-devel-ubuntu24.04 AS base

# Set noninteractive installation
ENV DEBIAN_FRONTEND=noninteractive

# Set CUDA environment variables
ENV PYTORCH_VERSION=2.9.0
ENV PYTORCH_CU_TAG=cu130
ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
ENV TORCH_CUDA_ARCH_LIST="8.9;10.0;12.0"

# System dependencies for build
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-venv \
    python3-dev \
    python3-pip \
    build-essential \
    ninja-build \
    ca-certificates
# Create an isolated virtual environment for all Python deps
RUN python3 -m venv /opt/venv
ENV VIRTUAL_ENV=/opt/venv
ENV PATH=/opt/venv/bin:${PATH}
ENV PIP_NO_CACHE_DIR=1

# Upgrade pip and build tooling
RUN python -m pip install --upgrade pip setuptools wheel ninja

# Install PyTorch
RUN pip install --no-cache-dir \
    torch==${PYTORCH_VERSION} \
    --index-url https://download.pytorch.org/whl/${PYTORCH_CU_TAG}

# Build context (pruned via .dockerignore)
WORKDIR /build
COPY . .

# Install the main package into the venv
RUN pip install --no-cache-dir .

# Build and install fused ops into the same venv
RUN cd fused_ops && python setup.py build_ext && python setup.py install && cd ..

# --------------------------
# Minimal runtime image
# --------------------------
FROM nvidia/cuda:${CUDA_TAG}-runtime-ubuntu24.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Only runtime necessities
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    ca-certificates \
    git

# Bring the prebuilt virtualenv from the base stage
COPY --from=base /opt/venv /opt/venv

# Use the venv by default
ENV PATH=/opt/venv/bin:${PATH}
ENV VIRTUAL_ENV=/opt/venv

WORKDIR /app

# Default: print versions (override with your own CMD/ENTRYPOINT)
ENTRYPOINT ["/bin/bash"]
