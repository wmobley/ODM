ARG CUDA_IMAGE=nvidia/cuda:12.8.1-devel-ubuntu24.04

FROM ${CUDA_IMAGE} AS gpu-deps

ENV DEBIAN_FRONTEND=noninteractive \
    CUDA_HOME="/usr/local/cuda" \
    CUDA_TOOLKIT_ROOT_DIR="/usr/local/cuda" \
    PATH="/code/venv/bin:/usr/local/cuda/bin:$PATH" \
    PYTHONPATH="/code/SuperBuild/install/lib/python3.12/dist-packages:/code/SuperBuild/install/bin/opensfm" \
    LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH:/code/SuperBuild/install/lib"

WORKDIR /code

COPY configure_gpu.sh configure.sh requirements.txt ./
COPY snap ./snap
COPY docker ./docker

RUN GPU_INSTALL=YES bash configure_gpu.sh installreqs
