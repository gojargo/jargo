# jargo developer commands.
#
# The Go toolchain is the build system. This file holds only the recipes that
# are more than a single command, so nothing here restates what AGENTS.md
# already gives you as a one-liner (`go build ./...`, `go test ./...`,
# `golangci-lint run`). Run `make help` for the list.
#
# The GitHub Actions workflows call these same targets wherever the two overlap,
# so a docs link failure or a vulnerability finding can be reproduced locally.
# Steps that talk to GitHub's own infrastructure (SARIF upload, Codecov, keyless
# signing, Pages deploys, multi-arch buildx) stay in the workflows as actions.

# Make runs each recipe line in /bin/sh with no errexit and no pipefail, which
# would let a failing `cmd | tee` report success. CI gates on these recipes, so
# make the shell strict.
SHELL := /usr/bin/env bash
.SHELLFLAGS := -eu -o pipefail -c
.DEFAULT_GOAL := help

GO ?= go

# `go install` writes to GOBIN when it is set and to GOPATH/bin otherwise. Put
# whichever it is on PATH so `go generate` finds the protoc plugins that
# `make tools-proto` puts there.
GO_BIN := $(shell $(GO) env GOBIN)
ifeq ($(GO_BIN),)
GO_BIN := $(shell $(GO) env GOPATH)/bin
endif
export PATH := $(GO_BIN):$(PATH)

# Tool versions, pinned so CI and a developer machine run the same thing. The
# two protoc plugin pins must match the versions stamped into the headers of the
# generated *.pb.go files, or `make generate-check` reports a false stale.
BUF_VERSION ?= v1.73.0
PROTOC_GEN_GO_VERSION ?= v1.36.12
PROTOC_GEN_GO_GRPC_VERSION ?= v1.6.2
GOVULNCHECK_VERSION ?= v1.8.0
GITLEAKS_IMAGE ?= ghcr.io/gitleaks/gitleaks:latest
SEMGREP_IMAGE ?= semgrep/semgrep:1.179.0
SEMGREP_SARIF ?= semgrep.sarif

COVERPROFILE ?= coverage.out

# Coverage is measured across the whole module rather than package by package,
# because a package is often covered by the tests of the one that drives it: the
# job context by the worker suite, the adapters by the services that use them.
# Measured package by package that code reads as untested, which sends people
# writing tests for behaviour that is already covered.
#
# The cost is that every test binary then instruments every package, so the raw
# profile repeats each block once per binary and runs past 100MB. COVER_MERGE
# folds the repeats back together, keeping the highest count each block was
# reached with, which is the union of what the whole run covered.
COVER_MERGE = awk '/^mode:/ { if (!m) { print; m = 1 } next } \
	{ k = $$1 " " $$2 } \
	!(k in c) { c[k] = $$3; o[++n] = k; next } \
	$$3 > c[k] { c[k] = $$3 } \
	END { for (i = 1; i <= n; i++) print o[i], c[o[i]] }'

# Native runtimes loaded at run time through purego. NATIVE_DIR is a prefix
# inside the repo so nothing here needs root; docker/runtime.Dockerfile installs
# the same two libraries into /usr/local for the container image.
#
# The cgo binding negotiates its API version with whatever runtime it finds, so
# this is not a floor the build has to clear. It tracks the release the binding
# itself ships in, which keeps the two halves of the same project in step.
ORT_VERSION ?= 1.30.0
ORT_ARCH ?= $(if $(filter aarch64 arm64,$(shell uname -m)),aarch64,x64)
NATIVE_DIR ?= $(CURDIR)/.native

##@ General

.PHONY: help
help: ## Print the available targets
	@awk 'BEGIN { FS = ":.*##"; printf "\nUsage: make <target>\n" } \
		/^[a-zA-Z0-9_-]+:.*?##/ { printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2 } \
		/^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) }' $(MAKEFILE_LIST)
	@echo

# Lets a workflow read a pin from here rather than keeping a second copy of it,
# for example `make -s print-HUGO_VERSION` in the docs workflow.
.PHONY: print-%
print-%:
	@echo "$($*)"

##@ Test

.PHONY: test
test: ## Run the tests with the race detector, as CI does
	$(GO) test -race ./...

.PHONY: cover
cover: ## Run the tests and write the coverage profile CI uploads
	$(GO) test -race -coverpkg=./... -coverprofile=$(COVERPROFILE).raw ./...
	@$(COVER_MERGE) $(COVERPROFILE).raw > $(COVERPROFILE)
	@rm -f $(COVERPROFILE).raw
	@$(GO) tool cover -func=$(COVERPROFILE) | tail -1
	@echo
	@echo "Codecov applies the ignore: rules in codecov.yml (examples/, generated"
	@echo "protobuf), so the reported total reads roughly 9 points above this one."
	@echo "Break it down with 'make cover-func' or 'make cover-html'."

.PHONY: require-coverprofile
require-coverprofile:
	@test -f $(COVERPROFILE) || { \
		echo "no $(COVERPROFILE) yet, run: make cover" >&2; \
		exit 1; \
	}

# Kept off `cover` itself: the per-function report is around 1600 lines, which
# is worth asking for and not worth printing on every CI run.
.PHONY: cover-func
cover-func: require-coverprofile ## Break the coverage profile down by function
	$(GO) tool cover -func=$(COVERPROFILE)

.PHONY: cover-html
cover-html: require-coverprofile ## Open the coverage profile in a browser
	$(GO) tool cover -html=$(COVERPROFILE)

##@ Build

.PHONY: build-matrix
build-matrix: ## Compile with cgo off and on
	@echo "==> cgo-free (the default build)"
	CGO_ENABLED=0 $(GO) build ./...
	@echo "==> cgo enabled"
	CGO_ENABLED=1 $(GO) build ./...

##@ Benchmarks

# The benchmark suite is a module of its own (benchmarks/go.mod), so the root
# `go build ./...` and `go vet ./...` never reach it. Without this check a change
# to the library's API could leave the suite broken until someone next tried to
# run it.
BENCH_DIR := benchmarks

.PHONY: bench-check
bench-check: ## Vet and build the benchmark suite against the library in the tree
	$(GO) -C $(BENCH_DIR) vet ./...
	$(GO) -C $(BENCH_DIR) build ./...

# The comparison runs each side in its own container, one after the other, both
# pinned to the same CPUs (BENCH_CPUSET, a cpuset list such as 2,3 or 0-3) and
# held to the same memory with no swap (BENCH_MEM). Pinning with --cpuset-cpus
# rather than --cpus matters for latency: a CFS quota lets a container run on
# every core in short bursts and then throttles it, which shows up as tail
# latency that belongs to the quota, not the framework. GOMAXPROCS is set to the
# number of CPUs in the set.
#
# The defaults suit a laptop: two cores and the quick profile. CI runs the full
# profile (BENCH_PROFILE=full) on a cpuset sized for its runner.
BENCH_CPUSET ?= 2,3
BENCH_MEM ?= 1g
BENCH_PROFILE ?= quick
BENCH_COUNT ?= 6
BENCH_RUN ?=
BENCH_OUT ?= $(CURDIR)/$(BENCH_DIR)/out
BENCH_PKGS ?= ./...
BENCH_PROCS = $(shell echo '$(BENCH_CPUSET)' | tr ',' '\n' | \
	awk -F- '{ n += (NF == 2 ? $$2 - $$1 + 1 : 1) } END { print n }')
# The compile inside the jargo image is held to this many CPUs; empty lifts it.
BENCH_BUILD_PROCS ?= 2
BENCH_JARGO_IMAGE ?= jargo-bench-jargo:local
BENCH_PIPECAT_IMAGE ?= jargo-bench-pipecat:local
BENCH_DOCKER_RUN = docker run --rm --cpuset-cpus=$(BENCH_CPUSET) \
	--memory=$(BENCH_MEM) --memory-swap=$(BENCH_MEM) \
	-v "$(CURDIR)/$(BENCH_DIR)/scenarios:/scenarios:ro"
BENCH_ARGS = run -count $(BENCH_COUNT) -run '$(BENCH_RUN)' /scenarios/$(BENCH_PROFILE).json

# jargo's own testing.B benchmarks, throttled so they can run on a machine that
# is in use. Compare two commits by running this on each and passing the two
# files to `go -C benchmarks tool benchstat`.
.PHONY: bench
bench: ## Run jargo's own benchmarks into benchmarks/out/bench.txt
	@mkdir -p $(BENCH_OUT)
	nice -n 19 env GOMAXPROCS=2 $(GO) -C $(BENCH_DIR) test -run '^$$' -bench . \
		-benchtime 1s -count 6 -p 1 $(BENCH_PKGS) | tee $(BENCH_OUT)/bench.txt

.PHONY: bench-images
bench-images: ## Build the two comparison images, jargo's and the Python side's
	nice -n 19 docker build -f $(BENCH_DIR)/Dockerfile \
		--build-arg FRAMEWORK_VERSION="$$(git describe --always --dirty 2>/dev/null || echo unknown)" \
		--build-arg GOMAXPROCS=$(BENCH_BUILD_PROCS) \
		-t $(BENCH_JARGO_IMAGE) .
	nice -n 19 docker build -f $(BENCH_DIR)/pipecat/Dockerfile \
		-t $(BENCH_PIPECAT_IMAGE) $(BENCH_DIR)/pipecat

# The two sides run one after the other, never at once, so neither competes
# with the other for the cpuset. benchstat then sets them side by side; the
# keys that necessarily differ between them (the framework's version and the
# runtime) are ignored so the two land in the same table.
.PHONY: bench-compare
bench-compare: bench-images ## Run the scenario on both sides under equal limits and compare
	@mkdir -p $(BENCH_OUT)
	@echo "==> jargo: cpuset $(BENCH_CPUSET), $(BENCH_MEM), GOMAXPROCS=$(BENCH_PROCS), profile $(BENCH_PROFILE)"
	$(BENCH_DOCKER_RUN) -e GOMAXPROCS=$(BENCH_PROCS) $(BENCH_JARGO_IMAGE) $(BENCH_ARGS) \
		| tee $(BENCH_OUT)/jargo.txt
	@echo "==> pipecat: cpuset $(BENCH_CPUSET), $(BENCH_MEM), profile $(BENCH_PROFILE)"
	$(BENCH_DOCKER_RUN) $(BENCH_PIPECAT_IMAGE) $(BENCH_ARGS) | tee $(BENCH_OUT)/pipecat.txt
	$(GO) -C $(BENCH_DIR) tool benchstat -col framework -ignore framework_version,runtime \
		$(BENCH_OUT)/jargo.txt $(BENCH_OUT)/pipecat.txt | tee $(BENCH_OUT)/compare.txt

# The Python side's lock is resolved inside the image it is installed in, so it
# holds the wheels Python 3.12 on Linux actually takes. The image is read from
# the Dockerfile, which Dependabot keeps current, rather than pinned twice.
PIP_TOOLS_VERSION ?= 7.6.2
BENCH_PYTHON_IMAGE = $(shell sed -n 's/^FROM \(python:[^ ]*\).*/\1/p' $(BENCH_DIR)/pipecat/Dockerfile)

.PHONY: bench-lock
bench-lock: ## Re-resolve the Python side's hashed requirements from requirements.in
	docker run --rm --cpuset-cpus=$(BENCH_CPUSET) --memory=$(BENCH_MEM) \
		-u "$$(id -u):$$(id -g)" -e HOME=/tmp \
		-v "$(CURDIR)/$(BENCH_DIR)/pipecat:/lock" -w /lock $(BENCH_PYTHON_IMAGE) \
		sh -c 'pip install --quiet --user pip-tools==$(PIP_TOOLS_VERSION) && \
			python -m piptools compile --quiet --generate-hashes --allow-unsafe \
			--strip-extras --output-file requirements.txt requirements.in'

##@ Generate

.PHONY: tools-proto
tools-proto: ## Install buf and the protoc plugins at their pinned versions
	$(GO) install github.com/bufbuild/buf/cmd/buf@$(BUF_VERSION)
	$(GO) install google.golang.org/protobuf/cmd/protoc-gen-go@$(PROTOC_GEN_GO_VERSION)
	$(GO) install google.golang.org/grpc/cmd/protoc-gen-go-grpc@$(PROTOC_GEN_GO_GRPC_VERSION)

.PHONY: generate
generate: tools-proto ## Regenerate the protobuf code from the .proto files
	$(GO) generate ./...

# Every directory holding checked-in generated protobuf code.
GENERATED_PROTO_DIRS := provider/nvidia/internal/rivapb \
	transport/wsserver/protobuf/internal/framepb

.PHONY: generate-check
generate-check: generate ## Fail if the checked-in generated code is stale
	@if ! git diff --quiet -- $(GENERATED_PROTO_DIRS); then \
		echo "generated protobuf is out of date, commit the result of: make generate" >&2; \
		git --no-pager diff --stat -- $(GENERATED_PROTO_DIRS) >&2; \
		exit 1; \
	fi

##@ Security

.PHONY: vuln
vuln: ## Report vulnerabilities jargo's code actually reaches
	$(GO) install golang.org/x/vuln/cmd/govulncheck@$(GOVULNCHECK_VERSION)
	govulncheck ./...

.PHONY: secrets
secrets: ## Scan the checked-out tree for secrets, as the security workflow does
	docker run --rm -v "$(CURDIR):/repo" $(GITLEAKS_IMAGE) \
		dir /repo --redact --no-banner --verbose

# use-tls is excluded: it flags every plain http.ListenAndServe, and the only
# calls in the tree are the example bots' local development servers. The
# library starts no HTTP server of its own, so TLS is the application's choice.
SEMGREP_EXCLUDE_RULES ?= go.lang.security.audit.net.use-tls.use-tls

.PHONY: sast
sast: ## Run the Semgrep CE static analysis and write its SARIF report
	docker run --rm -u "$$(id -u):$$(id -g)" -e HOME=/tmp -v "$(CURDIR):/src" -w /src $(SEMGREP_IMAGE) \
		semgrep scan --config p/golang --metrics=off \
		$(addprefix --exclude-rule=,$(SEMGREP_EXCLUDE_RULES)) \
		--sarif-output=$(SEMGREP_SARIF) --text

##@ Docs

# The site at https://gojargo.dev (repository gojargo/gojargo.dev) imports docs/
# at build time; nothing here builds it. What stays here is the check that the
# pages' relative links resolve, which `go test ./...` also runs in CI.
.PHONY: docs-check
docs-check: ## Fail on any relative link in docs/ or the README that points at nothing
	go test -run '^TestDocsLinks$$' -count=1 .

##@ Native dependencies

# These follow docker/runtime.Dockerfile, which is Debian and Linux only, so the
# host install is too. On another system install the same two libraries by hand:
# the ONNX Runtime and RNNoise.
.PHONY: require-linux
require-linux:
	@[[ "$$(uname -s)" == "Linux" ]] || { \
		echo "this target is Linux only, see the comment above it in the Makefile" >&2; \
		exit 1; \
	}

.PHONY: deps-onnx
deps-onnx: require-linux ## Fetch the ONNX Runtime for VAD and turn detection
	@mkdir -p $(NATIVE_DIR)
	curl -fsSL -o $(NATIVE_DIR)/ort.tgz \
		"https://github.com/microsoft/onnxruntime/releases/download/v$(ORT_VERSION)/onnxruntime-linux-$(ORT_ARCH)-$(ORT_VERSION).tgz"
	tar -xzf $(NATIVE_DIR)/ort.tgz -C $(NATIVE_DIR) --strip-components=1
	@rm -f $(NATIVE_DIR)/ort.tgz
	@echo
	@echo "export JARGO_ONNXRUNTIME_LIB=$(NATIVE_DIR)/lib/libonnxruntime.so"

.PHONY: deps-rnnoise
deps-rnnoise: require-linux ## Build RNNoise from source for the optional denoiser
	@mkdir -p $(NATIVE_DIR)/src
	rm -rf $(NATIVE_DIR)/src/rnnoise
	git clone --depth 1 https://github.com/xiph/rnnoise $(NATIVE_DIR)/src/rnnoise
	cd $(NATIVE_DIR)/src/rnnoise \
		&& ./autogen.sh \
		&& ./configure --disable-static --prefix=$(NATIVE_DIR) \
		&& make -j"$$(nproc)" \
		&& make install
	@echo
	@echo "export JARGO_RNNOISE_LIB=$(NATIVE_DIR)/lib/librnnoise.so.0"

##@ Packaging

# These build for the host platform only. The docker workflow publishes the real
# images multi-arch through buildx with tags from docker/metadata-action, and
# the release workflow signs artifacts with keyless cosign, neither of which is
# reproducible locally. Treat both as smoke tests of the Dockerfiles.

.PHONY: image-build
image-build: ## Build the jargo-build base image for the host platform
	docker build -f docker/build.Dockerfile -t jargo-build:local .

.PHONY: image-runtime
image-runtime: ## Build the distroless runtime base image for the host platform
	docker build -f docker/runtime.Dockerfile --build-arg ORT_VERSION=$(ORT_VERSION) \
		-t jargo:local .

.PHONY: release-snapshot
release-snapshot: ## Build the release artifacts without tagging or publishing
	@command -v goreleaser >/dev/null || { \
		echo "goreleaser is required: https://goreleaser.com/install/" >&2; \
		exit 1; \
	}
	goreleaser release --snapshot --clean

.PHONY: clean
clean: ## Remove build output, coverage profiles and the fetched native libraries
	rm -rf $(NATIVE_DIR) dist $(WEBSITE_DIR)/public $(WEBSITE_DIR)/resources/_gen
	rm -f $(COVERPROFILE)
