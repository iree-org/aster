# Build a portable LLVM package for Linux using a manylinux_2_28 base image.
# Binaries produced here depend only on glibc >= 2.28 and are suitable for
# distribution across modern Linux distros.
#
# Select architecture via --build-arg:
#   docker build --build-arg MANYLINUX_IMAGE=quay.io/pypa/manylinux_2_28_x86_64 ...
#   docker build --build-arg MANYLINUX_IMAGE=quay.io/pypa/manylinux_2_28_aarch64 ...
ARG MANYLINUX_IMAGE=quay.io/pypa/manylinux_2_28_x86_64
# hadolint ignore=DL3006
FROM ${MANYLINUX_IMAGE}

# Install clang (via the llvm-toolset module, AlmaLinux 8) plus build tools.
# patchelf is used after install to set portable RPATHs on any shared objects.
RUN dnf module enable -y llvm-toolset && \
    dnf install -y \
        clang \
        clang-devel \
        lld \
        cmake \
        git \
        python3-pip \
    && dnf clean all && \
    pip3 install ninja patchelf

# Copy the packaging scripts into the image so they are available at build time
# and when running the container without a bind-mounted source tree.
COPY . /pkgs/

ENTRYPOINT ["/bin/bash", "/pkgs/linux/entrypoint.sh"]
