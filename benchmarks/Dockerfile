# syntax=docker/dockerfile:1
#
# jargo side of the benchmark comparison: jbench, built from the library in this
# tree with the toolchain of docker/build.Dockerfile, on a distroless base. The
# context is the repository root, since the module builds against ../:
#
#   docker build -f benchmarks/Dockerfile \
#     --build-arg FRAMEWORK_VERSION="$(git describe --always --dirty)" \
#     -t jargo-bench-jargo .
#
# Run it with the scenarios mounted, under the same CPU and memory limits as
# the Python image and with GOMAXPROCS matching the CPUs it is given (see the
# bench-compare target in the Makefile):
#
#   docker run --rm --cpuset-cpus=2,3 --memory=1g -e GOMAXPROCS=2 \
#     -v "$PWD/benchmarks/scenarios:/scenarios:ro" jargo-bench-jargo \
#     run -count 6 /scenarios/quick.json
#
# The root .dockerignore leaves benchmarks/ out of every other image, so this
# one has its own, Dockerfile.dockerignore beside it.

FROM golang:1.27-bookworm@sha256:d9b6a1beb9474adc2a1ce842573639b44b2a0dd49b6669f51d3821baf9ec37c9 AS build

ENV CGO_ENABLED=0
WORKDIR /src
COPY go.mod go.sum ./
COPY benchmarks/go.mod benchmarks/go.sum ./benchmarks/
RUN go -C benchmarks mod download
COPY . .
# The build context carries no .git, so the revision is passed in rather than
# read from the VCS.
ARG FRAMEWORK_VERSION=unknown
# Caps the compiler's parallelism when set, for a machine that is in use; empty
# leaves it to the toolchain.
ARG GOMAXPROCS=
RUN go -C benchmarks build -trimpath -buildvcs=false \
      -ldflags "-s -w -X main.frameworkVersion=${FRAMEWORK_VERSION}" \
      -o /out/jbench ./cmd/jbench

# Built without cgo, the binary still links libc dynamically (purego reaches
# dlopen through it), so the base is distroless/cc, as for docker/runtime.Dockerfile.
FROM gcr.io/distroless/cc-debian12:nonroot@sha256:9dac0a79194e45a7da0158a9c6da57b217585af0786db3845d1f0ec1a0dd182f

COPY --from=build /out/jbench /usr/local/bin/jbench
ENTRYPOINT ["/usr/local/bin/jbench"]
