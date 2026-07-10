# syntax=docker/dockerfile:1
#
# jargo runtime base image.
#
# A distroless image carrying only the native shared libraries a compiled jargo
# bot needs at run time — libsoxr, libopus and the ONNX Runtime (loaded for VAD
# and turn detection) — on top of distroless/cc (glibc + libstdc++, which the
# C++ ONNX Runtime needs) with CA certificates. There is no shell and no package
# manager, so the attack surface is small, and the default user is non-root.
#
# Ship a bot by copying a jargo binary (built with the jargo-build base) in:
#
#   FROM gojargo/jargo
#   COPY --from=build /out/bot /usr/local/bin/bot
#   ENTRYPOINT ["/usr/local/bin/bot"]

# ---- collect the native runtime libraries from Debian + the ONNX release ----
FROM debian:bookworm-slim@sha256:60eac759739651111db372c07be67863818726f754804b8707c90979bda511df AS libs
ARG ORT_VERSION=1.26.0
ARG TARGETARCH=amd64
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl libsoxr0 libopus0 \
    && rm -rf /var/lib/apt/lists/*
RUN set -eux; \
    case "${TARGETARCH}" in \
      amd64) ort_arch=x64 ;; \
      arm64) ort_arch=aarch64 ;; \
      *) echo "unsupported TARGETARCH=${TARGETARCH}" >&2; exit 1 ;; \
    esac; \
    curl -fsSL -o /tmp/ort.tgz \
      "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-linux-${ort_arch}-${ORT_VERSION}.tgz"; \
    mkdir -p /opt/ort && tar -xzf /tmp/ort.tgz -C /opt/ort --strip-components=1; \
    cp -L /opt/ort/lib/libonnxruntime.so /usr/local/lib/libonnxruntime.so

# ---- distroless runtime ----
FROM gcr.io/distroless/cc-debian12:nonroot@sha256:b0ae8e989418b458e0f25489bc3be523718938a2b70864cc0f6a00af1ddbd985

# The linked libs go in the default multiarch search path so the dynamic linker
# resolves them — distroless has no ldconfig. libsoxr pulls in libgomp (OpenMP),
# which distroless/cc does not ship, so copy it too. (amd64 only for now; an
# arm64 build would copy from /usr/lib/aarch64-linux-gnu instead.)
COPY --from=libs /usr/lib/x86_64-linux-gnu/libsoxr.so.0* /usr/lib/x86_64-linux-gnu/
COPY --from=libs /usr/lib/x86_64-linux-gnu/libgomp.so.1* /usr/lib/x86_64-linux-gnu/
COPY --from=libs /usr/lib/x86_64-linux-gnu/libopus.so.0* /usr/lib/x86_64-linux-gnu/

# The ONNX Runtime is loaded at run time by the explicit path in this env var.
COPY --from=libs /usr/local/lib/libonnxruntime.so /usr/local/lib/libonnxruntime.so
ENV JARGO_ONNXRUNTIME_LIB=/usr/local/lib/libonnxruntime.so

# The distroless base already defaults to the non-root user; state it
# explicitly so it is enforced and visible to scanners.
USER nonroot

# jargo example bots serve on :8080 (informational; downstream can override).
EXPOSE 8080
