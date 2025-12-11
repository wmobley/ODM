FROM ubuntu:24.04 AS builder

# Env variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONPATH="$PYTHONPATH:/code/SuperBuild/install/lib/python3.12/dist-packages:/code/SuperBuild/install/bin/opensfm" \
    LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/code/SuperBuild/install/lib"

# Prepare directories
WORKDIR /code

# Copy only build scripts and manifests first to maximize cache reuse
COPY configure.sh configure.py ./
COPY SuperBuild ./SuperBuild
COPY opendm/context.py ./opendm/context.py
COPY requirements.txt .

# System deps needed early (GDAL headers/tools for python bindings)
RUN --mount=type=cache,target=/var/cache/apt \
    --mount=type=cache,target=/var/lib/apt/lists \
    apt-get update -y && \
    apt-get install -y gdal-bin libgdal-dev && \
    rm -rf /var/lib/apt/lists/*

# Run the build with cache mounts for faster rebuilds
RUN --mount=type=cache,target=/var/cache/apt \
    --mount=type=cache,target=/var/lib/apt/lists \
    --mount=type=cache,target=/root/.cache/pip \
    --mount=type=cache,target=/root/.ccache \
    PORTABLE_INSTALL=YES bash configure.sh install

# Now bring in the remaining source (smaller cache blast radius)
COPY . ./

# (Tests skipped in CI Docker build to reduce duration; run separately if needed)
ENV PATH="/code/venv/bin:$PATH"

# Clean Superbuild
RUN bash configure.sh clean

### END Builder

### Use a second image for the final asset to reduce the number and
# size of the layers.
FROM ubuntu:24.04

# Env variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONPATH="$PYTHONPATH:/code/SuperBuild/install/lib/python3.12/dist-packages:/code/SuperBuild/install/bin/opensfm" \
    LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/code/SuperBuild/install/lib" \
    PDAL_DRIVER_PATH="/code/SuperBuild/install/bin"

WORKDIR /code

# Copy everything we built from the builder
COPY --from=builder /code /code

ENV PATH="/code/venv/bin:$PATH"

# Install shared libraries that we depend on via APT, but *not*
# the -dev packages to save space!
# Also run a smoke test on ODM and OpenSfM
RUN bash configure.sh installruntimedepsonly \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* \
  && bash run.sh --help \
  && bash -c "eval $(python3 /code/opendm/context.py) && python3 -c 'from opensfm import io, pymap'"

# Entry point
ENTRYPOINT ["python3", "/code/run.py"]
