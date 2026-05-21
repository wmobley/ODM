ARG CUDA_IMAGE=nvidia/cuda:12.8.1-devel-ubuntu24.04

FROM ${CUDA_IMAGE} AS builder

ARG ODM_BUILD_PROCESSES=4

# Env variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONPATH="/code/SuperBuild/install/lib/python3.12/dist-packages:/code/SuperBuild/install/bin/opensfm" \
    LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/code/SuperBuild/install/lib"

# Prepare directories
WORKDIR /code

# Copy everything
COPY . ./

# Run the build
RUN PORTABLE_INSTALL=YES GPU_INSTALL=YES bash configure.sh install ${ODM_BUILD_PROCESSES}

# Tests are skipped in CI Docker builds to keep publish builds focused on
# producing the runtime image. The final stage still runs ODM/OpenSfM smoke
# checks after runtime dependencies are installed.
ENV PATH="/code/venv/bin:$PATH"

# Clean Superbuild
RUN bash configure.sh clean

### END Builder

### Use a second image for the final asset to reduce the number and
# size of the layers.
FROM ${CUDA_IMAGE}

# Env variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONPATH="/code/SuperBuild/install/lib/python3.12/dist-packages:/code/SuperBuild/install/bin/opensfm" \
    LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/code/SuperBuild/install/lib" \
    PDAL_DRIVER_PATH="/code/SuperBuild/install/bin"

WORKDIR /code

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
RUN bash configure.sh installruntimedepsonly \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* \
  && find /code/SuperBuild/install/bin/opensfm/opensfm -name 'pypopsift*.so' -print -quit | grep -q . \
  && bash run.sh --help \
  && bash -c "eval $(python3 /code/opendm/context.py) && python3 -c 'from opensfm import io, pymap'"

# Entry point
ENTRYPOINT ["python3", "/code/run.py"]
