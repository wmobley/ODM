# syntax=docker/dockerfile:1.6

ARG CUDA_IMAGE=nvidia/cuda:12.8.1-devel-ubuntu24.04

FROM ${CUDA_IMAGE} AS base

ENV DEBIAN_FRONTEND=noninteractive \
    CUDA_HOME="/usr/local/cuda" \
    CUDA_TOOLKIT_ROOT_DIR="/usr/local/cuda" \
    PATH="/usr/local/cuda/bin:$PATH" \
    PYTHONPATH="/code/SuperBuild/install/lib/python3.12/dist-packages:/code/SuperBuild/install/bin/opensfm" \
    LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH:/code/SuperBuild/install/lib"

WORKDIR /code

FROM base AS cuda-smoke

RUN --mount=type=cache,target=/var/cache/apt \
    apt-get update -y \
 && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    cmake \
    git \
    lsb-release \
    pkg-config \
    python3 \
    python3-dev \
    python3-pip \
    python3-setuptools \
    python3-venv \
 && rm -rf /var/lib/apt/lists/*

RUN nvcc --version \
 && cmake --version \
 && python3 --version \
 && test -d /usr/local/cuda/include \
 && test -f /usr/local/cuda/lib64/libcudart.so

FROM cuda-smoke AS odm-deps

COPY configure.sh requirements.txt ./
COPY snap ./snap

RUN --mount=type=cache,target=/root/.cache/pip \
    GPU_INSTALL=YES bash configure.sh installreqs

FROM odm-deps AS odm-superbuild-pypopsift

ARG ODM_BUILD_PROCESSES=2
ARG GPUCACHEBUST=0

COPY SuperBuild ./SuperBuild

RUN echo "GPUCACHEBUST=${GPUCACHEBUST}" \
 && export GPU_INSTALL=YES \
 && cd /code/SuperBuild \
 && mkdir -p build install/bin/opensfm/opensfm \
 && cd build \
 && cmake .. -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda \
 && cmake --build . --target pypopsift -- -j${ODM_BUILD_PROCESSES} \
 && (find /code/SuperBuild/install -name 'pypopsift*.so' -print -quit | grep -q . \
     || (echo "ERROR: pypopsift*.so was not installed" \
         && find /code/SuperBuild -iname '*popsift*' -print | sort \
         && exit 1)) \
 && /code/venv/bin/python3 -c 'import sys; sys.path.insert(0, "/code/SuperBuild/install/bin/opensfm/opensfm"); import pypopsift; print("pypopsift import ok")'
