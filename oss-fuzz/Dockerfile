# OSS-Fuzz packaging for the high-priority Zig fuzz harnesses.
#
# Repository-local OSS-Fuzz integration. Mirrors the upstream
# OSS-Fuzz `projects/<name>/Dockerfile` layout but is **not** an
# upstream submission. Building this image and running
# `compile`/`build.sh` produces libFuzzer-driven binaries for
# `fuzz-loader`, `fuzz-component-loader`, and `fuzz-canon` so a
# maintainer can run coverage-guided fuzzing locally.
#
# See `tests/fuzz/OSS_FUZZ.md` for the readiness analysis and
# `oss-fuzz/build.sh` for the build steps.

FROM gcr.io/oss-fuzz-base/base-builder:v1

# Install Zig 0.16.0 with checksum verification. The download index
# at https://ziglang.org/download/index.json pins this checksum.
ENV ZIG_VERSION=0.16.0
ENV ZIG_TARBALL=zig-x86_64-linux-${ZIG_VERSION}.tar.xz
ENV ZIG_SHA256=70e49664a74374b48b51e6f3fdfbf437f6395d42509050588bd49abe52ba3d00
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends curl ca-certificates xz-utils; \
    rm -rf /var/lib/apt/lists/*; \
    curl -fsSL "https://ziglang.org/download/${ZIG_VERSION}/${ZIG_TARBALL}" -o /tmp/zig.tar.xz; \
    echo "${ZIG_SHA256}  /tmp/zig.tar.xz" | sha256sum -c -; \
    mkdir -p /opt; \
    tar -C /opt -xJf /tmp/zig.tar.xz; \
    rm /tmp/zig.tar.xz; \
    ln -s "/opt/zig-x86_64-linux-${ZIG_VERSION}/zig" /usr/local/bin/zig; \
    zig version

# OSS-Fuzz convention: the project source is cloned into $SRC.
RUN git clone --depth 1 https://github.com/cataggar/wamr.git "$SRC/wamr"

WORKDIR $SRC/wamr
COPY build.sh $SRC/build.sh
