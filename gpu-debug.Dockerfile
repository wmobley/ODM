FROM nvidia/cuda:12.8.1-devel-ubuntu24.04

ARG DEBIAN_FRONTEND=noninteractive

ENV CUDA_HOME=/usr/local/cuda
ENV CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda
ENV PATH=/usr/local/cuda/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
ENV PORTABLE_INSTALL=YES
ENV GPU_INSTALL=YES
ENV ODM_BUILD_PROCESSES=1

WORKDIR /code

RUN apt-get update \
  && apt-get install -y --no-install-recommends \
      bash \
      ca-certificates \
      git \
      curl \
      wget \
      build-essential \
      cmake \
      ninja-build \
      pkg-config \
      python3 \
      python3-pip \
      python3-dev \
      python3-setuptools \
      python3-wheel \
      vim \
      nano \
      less \
      tree \
      file \
      findutils \
      grep \
      sed \
      procps \
      gdb \
      strace \
      lsb-release \
      software-properties-common \
  && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3 /usr/local/bin/python

COPY . /code

CMD ["/bin/bash"]
