ARG BASE_IMAGE=ubuntu:24.04
ARG CPU_DEPS_IMAGE=odm-cpu-deps:local

FROM ${CPU_DEPS_IMAGE} AS builder

ARG ODM_BUILD_PROCESSES=1
ARG CACHEBUST=0
ARG POTREECACHEBUST=0

ENV PATH="/code/venv/bin:$PATH"

WORKDIR /code

COPY . ./

RUN echo "CACHEBUST=${CACHEBUST}" \
  && PORTABLE_INSTALL=YES bash configure.sh buildsuper ${ODM_BUILD_PROCESSES}

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

RUN bash configure.sh clean

FROM ${BASE_IMAGE} AS runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONPATH="$PYTHONPATH:/code/SuperBuild/install/lib/python3.12/dist-packages:/code/SuperBuild/install/bin/opensfm" \
    LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/code/SuperBuild/install/lib" \
    PDAL_DRIVER_PATH="/code/SuperBuild/install/bin"

WORKDIR /code

COPY --from=builder /code /code

ENV PATH="/code/venv/bin:/code/SuperBuild/install/bin:$PATH"

RUN ln -sf /code/SuperBuild/install/bin/PotreeConverter /usr/local/bin/PotreeConverter \
  && bash configure.sh installruntimedepsonly \
  && apt-get update \
  && apt-get install -y --no-install-recommends liblaszip8 \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* \
  && bash run.sh --help \
  && bash -c "eval $(python3 /code/opendm/context.py) && python3 -c 'from opensfm import io, pymap'"

ENTRYPOINT ["python3", "/code/run.py"]
