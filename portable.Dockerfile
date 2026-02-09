FROM ubuntu:24.04 AS builder

ARG ODM_BUILD_PROCESSES=4
ARG POTREECACHEBUST=0

# Env variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONPATH="$PYTHONPATH:/code/SuperBuild/install/lib/python3.12/dist-packages:/code/SuperBuild/install/bin/opensfm" \
    LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/code/SuperBuild/install/lib"

# Prepare directories
WORKDIR /code

# Copy everything
COPY . ./

# Run the build
RUN PORTABLE_INSTALL=YES bash configure.sh install ${ODM_BUILD_PROCESSES}

# (Tests skipped in CI Docker build to reduce duration; run separately if needed)
ENV PATH="/code/venv/bin:$PATH"

# Build and install PotreeConverter for point cloud tiling fallback
RUN set -eux; echo "POTREECACHEBUST=${POTREECACHEBUST}" \
  && apt-get update \
  && apt-get install -y --no-install-recommends \
    git cmake g++ make libtbb-dev libboost-all-dev liblaszip-dev libeigen3-dev \
  && rm -rf /var/lib/apt/lists/* \
  && if ! command -v PotreeConverter >/dev/null 2>&1; then \
       git clone --depth 1 --branch 1.7 https://github.com/potree/PotreeConverter.git /tmp/PotreeConverter \
       && python3 /code/docker/patch_potree.py /tmp/PotreeConverter \
       && cmake -S /tmp/PotreeConverter -B /tmp/PotreeConverter/build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_STANDARD=17 \
       && cmake --build /tmp/PotreeConverter/build -j"$(nproc)" \
       && POTREE_BIN="$(find /tmp/PotreeConverter/build -type f -name PotreeConverter -print -quit)" \
       && test -n "$POTREE_BIN" \
       && install -m 755 "$POTREE_BIN" /code/SuperBuild/install/bin/PotreeConverter \
       && test -x /code/SuperBuild/install/bin/PotreeConverter \
       && cp -R /tmp/PotreeConverter /code/PotreeConverter; \
     fi \
  && rm -rf /tmp/PotreeConverter

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

ENV PATH="/code/venv/bin:/code/SuperBuild/install/bin:$PATH"

# Install shared libraries that we depend on via APT, but *not*
# the -dev packages to save space!
# Also run a smoke test on ODM and OpenSfM
RUN ln -sf /code/SuperBuild/install/bin/PotreeConverter /usr/local/bin/PotreeConverter \
  && bash configure.sh installruntimedepsonly \
  && apt-get update \
  && apt-get install -y --no-install-recommends liblaszip8 \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* \
  && bash run.sh --help \
  && bash -c "eval $(python3 /code/opendm/context.py) && python3 -c 'from opensfm import io, pymap'"

# Entry point
ENTRYPOINT ["python3", "/code/run.py"]
