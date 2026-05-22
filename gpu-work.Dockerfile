ARG CUDA_IMAGE=nvidia/cuda:12.8.1-devel-ubuntu24.04
ARG GPU_DEPS_IMAGE=odm-gpu-deps:local

FROM ${GPU_DEPS_IMAGE} AS builder

ARG ODM_BUILD_PROCESSES=1
ARG ODM_GPU_PYPOPSIFT_ONLY=
ARG GPUCACHEBUST=0

ENV PATH="/code/venv/bin:$PATH"

WORKDIR /code

COPY . ./

RUN echo "GPUCACHEBUST=${GPUCACHEBUST}" \
  && PORTABLE_INSTALL=YES GPU_INSTALL=YES ODM_GPU_PYPOPSIFT_ONLY="${ODM_GPU_PYPOPSIFT_ONLY}" bash configure_gpu.sh buildsuper ${ODM_BUILD_PROCESSES} \
  && (find /code/SuperBuild/install -name 'pypopsift*.so' -print -quit | grep -q . \
      || (echo "ERROR: pypopsift was not installed by the GPU build" \
          && find /code/SuperBuild -iname '*popsift*' -print | sort \
          && exit 1))

RUN bash configure_gpu.sh clean

FROM ${CUDA_IMAGE} AS runtime

ARG ODM_GPU_PYPOPSIFT_ONLY=

ENV DEBIAN_FRONTEND=noninteractive \
    CUDA_HOME="/usr/local/cuda" \
    CUDA_TOOLKIT_ROOT_DIR="/usr/local/cuda" \
    PATH="/usr/local/cuda/bin:$PATH" \
    PYTHONPATH="/code/SuperBuild/install/lib/python3.12/dist-packages:/code/SuperBuild/install/bin/opensfm" \
    LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH:/code/SuperBuild/install/lib" \
    PDAL_DRIVER_PATH="/code/SuperBuild/install/bin"

WORKDIR /code

COPY --from=builder /code /code

ENV PATH="/code/venv/bin:$PATH"

RUN apt-get update -y \
 && apt-get install -y --no-install-recommends \
    ffmpeg \
    libtbb12 \
    libtbbmalloc2

RUN bash configure_gpu.sh installruntimedepsonly \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* \
  && (find /code/SuperBuild/install -name 'pypopsift*.so' -print -quit | grep -q . \
      || (echo "ERROR: pypopsift is missing from the runtime image" \
          && find /code/SuperBuild -iname '*popsift*' -print | sort \
          && exit 1)) \
  && if [ -z "$ODM_GPU_PYPOPSIFT_ONLY" ]; then \
       bash run.sh --help \
       && bash -c "eval $(python3 /code/opendm/context.py) && python3 -c 'from opensfm import io, pymap, pypopsift'"; \
     else \
       python3 -c 'from opensfm import pypopsift'; \
     fi

ENTRYPOINT ["python3", "/code/run.py"]
