ARG CUDA_IMAGE=nvidia/cuda:12.8.1-devel-ubuntu24.04

FROM ${CUDA_IMAGE} AS gpu-base

# Env variables
ENV DEBIAN_FRONTEND=noninteractive \
    CUDA_HOME="/usr/local/cuda" \
    CUDA_TOOLKIT_ROOT_DIR="/usr/local/cuda" \
    PATH="/usr/local/cuda/bin:$PATH" \
    PYTHONPATH="/code/SuperBuild/install/lib/python3.12/dist-packages:/code/SuperBuild/install/bin/opensfm" \
    LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH:/code/SuperBuild/install/lib"

# Prepare directories
WORKDIR /code

FROM gpu-base AS pypopsift-debug

ARG ODM_BUILD_PROCESSES=4
ARG GPUCACHEBUST=0

# Keep dependency installation cacheable while iterating on SuperBuild/CMake.
COPY configure_gpu.sh requirements.txt ./
COPY snap ./snap
RUN GPU_INSTALL=YES bash configure_gpu.sh installreqs

# Copy everything else after dependencies are installed.
COPY . ./

# Build only the CUDA SIFT extension. This target is for fast CI debugging;
# it avoids building the full ODM SuperBuild before proving PyPopSift works.
RUN echo "GPUCACHEBUST=${GPUCACHEBUST}" \
  && export GPU_INSTALL=YES \
  && cd /code/SuperBuild \
  && mkdir -p build install/bin/opensfm/opensfm \
  && cd build \
  && cmake .. \
  && cmake --build . --target pypopsift -- -j${ODM_BUILD_PROCESSES} \
  && (find /code/SuperBuild/install -name 'pypopsift*.so' -print -quit | grep -q . \
      || (echo "ERROR: pypopsift debug target did not install pypopsift*.so" \
          && find /code/SuperBuild -iname '*popsift*' -print | sort \
          && exit 1))

FROM gpu-base AS builder

ARG ODM_BUILD_PROCESSES=4
ARG GPUCACHEBUST=0
ARG ODM_GPU_PYPOPSIFT_ONLY=

# Copy everything
COPY . ./
# PyPopSift already writes the extension into the OpenSfM package directory via OUTPUT_DIR.
# The default CMake install target is broken in this path, so skip only the ExternalProject
# install step while preserving the actual build output.
# PyPopSift already writes the extension into the OpenSfM package directory.
# The default CMake install target is broken here, so skip only the install step.
RUN grep -q 'INSTALL_COMMAND ""' /code/SuperBuild/cmake/External-PyPopsift.cmake \
  || sed -i '/INSTALL_DIR ${SB_INSTALL_DIR}/a\    INSTALL_COMMAND ""' /code/SuperBuild/cmake/External-PyPopsift.cmake \
  && grep -n 'INSTALL_COMMAND\|INSTALL_DIR' /code/SuperBuild/cmake/External-PyPopsift.cmake

  
# Run the build
RUN echo "GPUCACHEBUST=${GPUCACHEBUST}" \
  && PORTABLE_INSTALL=YES GPU_INSTALL=YES ODM_GPU_PYPOPSIFT_ONLY=${ODM_GPU_PYPOPSIFT_ONLY} bash configure_gpu.sh install ${ODM_BUILD_PROCESSES} \
  && (find /code/SuperBuild/install -name 'pypopsift*.so' -print -quit | grep -q . \
      || (echo "ERROR: pypopsift was not installed by the GPU build" \
          && find /code/SuperBuild -iname '*popsift*' -print | sort \
          && exit 1))

# Tests are skipped in CI Docker builds to keep publish builds focused on
# producing the runtime image. The final stage still runs ODM/OpenSfM smoke
# checks after runtime dependencies are installed.
ENV PATH="/code/venv/bin:$PATH"

# Clean Superbuild
RUN bash configure_gpu.sh clean

### END Builder

### Use a second image for the final asset to reduce the number and
# size of the layers.
FROM gpu-base

ENV PDAL_DRIVER_PATH="/code/SuperBuild/install/bin"

# Copy everything we built from the builder
COPY --from=builder /code /code

ENV PATH="/code/venv/bin:$PATH"

RUN apt-get update -y \
 && apt-get install -y --no-install-recommends \
    ffmpeg \
    libtbb12 \
    libtbbmalloc2
# Install shared libraries that we depend on via APT, but *not*
# the -dev packages to save space!
# Also run a smoke test on ODM and OpenSfM
RUN bash configure_gpu.sh installruntimedepsonly \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* \
  && (find /code/SuperBuild/install -name 'pypopsift*.so' -print -quit | grep -q . \
      || (echo "ERROR: pypopsift is missing from the runtime image" \
          && find /code/SuperBuild -iname '*popsift*' -print | sort \
          && exit 1)) \
  && bash run.sh --help \
  && bash -c "eval $(python3 /code/opendm/context.py) && python3 -c 'from opensfm import io, pymap, pypopsift'"

# Entry point
ENTRYPOINT ["python3", "/code/run.py"]
