ARG BASE_IMAGE=ubuntu:24.04

FROM ${BASE_IMAGE} AS cpu-deps

ENV DEBIAN_FRONTEND=noninteractive \
    PATH="/code/venv/bin:$PATH" \
    PYTHONPATH="/code/SuperBuild/install/lib/python3.12/dist-packages:/code/SuperBuild/install/bin/opensfm" \
    LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/code/SuperBuild/install/lib"

WORKDIR /code

COPY configure.sh requirements.txt ./
COPY snap ./snap
COPY docker ./docker

RUN bash configure.sh installreqs
