# syntax=docker/dockerfile:1
#
# Pipecat side of the benchmark comparison: Python 3.12 (the version Pipecat
# recommends and tests on) with pipecat-ai installed from a lock that pins every
# package by hash. Build it from the benchmarks directory:
#
#   docker build -f benchmarks/pipecat/Dockerfile -t jargo-bench-pipecat benchmarks/pipecat
#
# and run it with the scenarios mounted, under the same CPU and memory limits
# as the jargo image (see the bench-compare target in the Makefile):
#
#   docker run --rm --cpuset-cpus=2,3 --memory=1g \
#     -v "$PWD/benchmarks/scenarios:/scenarios:ro" jargo-bench-pipecat \
#     run -count 6 /scenarios/quick.json

FROM python:3.15-slim@sha256:65a65fc7b30b06cb6df4930fd4650697cb23f4c2698ab81fc87e4b031bc36928

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /bench
COPY requirements.txt .
# Every package, transitive ones included, is pinned and checked by hash. One of
# them (docopt, through num2words) ships only as source, so source distributions
# cannot be refused outright; its hash is checked like the rest.
RUN pip install --require-hashes --no-deps -r requirements.txt
COPY bench.py .

USER nobody
# A benchmark run is a batch job that exits when it is done, with no service to
# probe.
HEALTHCHECK NONE
# -I keeps the environment and the working directory out of the import path.
ENTRYPOINT ["python", "-I", "/bench/bench.py"]
