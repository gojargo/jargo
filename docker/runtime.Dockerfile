# syntax=docker/dockerfile:1
#
# jargo runtime base image.
#
# A distroless image carrying the native shared libraries a compiled bot loads
# at run time: the ONNX Runtime (VAD + turn detection) and RNNoise (the optional
# denoiser), both dlopen'd via purego. It sits on distroless/cc (glibc +
# libstdc++, which the C++ ONNX Runtime needs) with CA certificates. There is no
# shell and no package manager, so the attack surface is small, and the default
# user is non-root.
#
# Ship a bot by copying a jargo binary (built with the jargo-build base) in:
#
#   FROM gojargo/jargo
#   COPY --from=build /out/bot /usr/local/bin/bot
#   ENTRYPOINT ["/usr/local/bin/bot"]

# ---- collect the native runtime libraries from Debian + the ONNX release ----
FROM debian:bookworm-slim@sha256:7c7b2c966bc9ee8cedfeef67e0e279108992c77681fa595db4a9d65c06ccc587 AS libs
ARG ORT_VERSION=1.30.0
# Where the ONNX Runtime comes from: github, Microsoft's release archive, or
# nuget, Microsoft's Microsoft.ML.OnnxRuntime package, whose libonnxruntime.so
# is the same file. nuget while the GitHub release is unavailable (#152); the
# package is checked against ORT_NUGET_SHA256, which changes with ORT_VERSION.
ARG ORT_SOURCE=nuget
ARG ORT_NUGET_SHA256=30df119bd9164eaa187666d93147448fdd4a9a1a3caf83f1505daf7af2d1c5d2
ARG TARGETARCH=amd64
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates curl wget unzip \
        git build-essential autoconf automake libtool \
    && rm -rf /var/lib/apt/lists/*
RUN set -eux; \
    case "${TARGETARCH}" in \
      amd64) ort_arch=x64; ort_rid=linux-x64 ;; \
      arm64) ort_arch=aarch64; ort_rid=linux-arm64 ;; \
      *) echo "unsupported TARGETARCH=${TARGETARCH}" >&2; exit 1 ;; \
    esac; \
    case "${ORT_SOURCE}" in \
      github) \
        curl -fsSL -o /tmp/ort.tgz \
          "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-linux-${ort_arch}-${ORT_VERSION}.tgz"; \
        mkdir -p /opt/ort && tar -xzf /tmp/ort.tgz -C /opt/ort --strip-components=1; \
        cp -L /opt/ort/lib/libonnxruntime.so /usr/local/lib/libonnxruntime.so ;; \
      nuget) \
        curl -fsSL -o /tmp/ort.nupkg \
          "https://api.nuget.org/v3-flatcontainer/microsoft.ml.onnxruntime/${ORT_VERSION}/microsoft.ml.onnxruntime.${ORT_VERSION}.nupkg"; \
        echo "${ORT_NUGET_SHA256}  /tmp/ort.nupkg" | sha256sum -c -; \
        unzip -q /tmp/ort.nupkg "runtimes/${ort_rid}/native/libonnxruntime.so" -d /opt/ort; \
        cp /opt/ort/runtimes/${ort_rid}/native/libonnxruntime.so /usr/local/lib/libonnxruntime.so ;; \
      *) echo "ORT_SOURCE must be github or nuget" >&2; exit 1 ;; \
    esac
# RNNoise is not packaged in Debian; build it from source so the purego denoiser
# can dlopen it at run time. Only the resulting .so is copied into the final image.
RUN set -eux; \
    git clone --depth 1 https://github.com/xiph/rnnoise /tmp/rnnoise; \
    cd /tmp/rnnoise; ./autogen.sh; ./configure --disable-static --prefix=/usr/local; \
    make -j"$(nproc)"; make install

# ---- distroless runtime ----
FROM gcr.io/distroless/cc-debian12:nonroot@sha256:9dac0a79194e45a7da0158a9c6da57b217585af0786db3845d1f0ec1a0dd182f

# Both libraries below are loaded by the explicit paths in the env vars, so they
# need no ldconfig (distroless has none). The ONNX Runtime needs only libstdc++
# and libgcc_s, which distroless/cc already ships.

# The ONNX Runtime is loaded at run time by the explicit path in this env var.
COPY --from=libs /usr/local/lib/libonnxruntime.so /usr/local/lib/libonnxruntime.so
ENV JARGO_ONNXRUNTIME_LIB=/usr/local/lib/libonnxruntime.so

# RNNoise for the optional purego denoiser, also loaded by explicit path.
COPY --from=libs /usr/local/lib/librnnoise.so.0* /usr/local/lib/
ENV JARGO_RNNOISE_LIB=/usr/local/lib/librnnoise.so.0

# The distroless base already defaults to the non-root user; state it
# explicitly so it is enforced and visible to scanners.
USER nonroot

# jargo example bots serve on :8080 (informational; downstream can override).
EXPOSE 8080
