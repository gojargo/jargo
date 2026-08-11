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
# would let a failing `hugo | tee` report success. CI gates on these recipes, so
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
BUF_VERSION ?= v1.72.0
PROTOC_GEN_GO_VERSION ?= v1.36.11
PROTOC_GEN_GO_GRPC_VERSION ?= v1.5.1
GOVULNCHECK_VERSION ?= v1.5.0
GITLEAKS_IMAGE ?= ghcr.io/gitleaks/gitleaks:latest

# Hugo is installed out of band. This is the single pin for it: the docs
# workflow provisions the runner from `make -s print-HUGO_VERSION`, and the
# require-hugo guard names it when the binary is missing locally.
HUGO_VERSION ?= 0.152.0
WEBSITE_DIR := website
HUGO_CHECK_LOG ?= /tmp/hugo-check.log
# Set by the docs workflow from the Pages configuration; empty for local builds,
# which fall back to the baseURL in website/hugo.toml.
HUGO_BASEURL ?=

COVERPROFILE ?= coverage.out

# Native runtimes loaded at run time through purego. NATIVE_DIR is a prefix
# inside the repo so nothing here needs root; docker/runtime.Dockerfile installs
# the same two libraries into /usr/local for the container image.
ORT_VERSION ?= 1.26.0
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
	$(GO) test -race -coverprofile=$(COVERPROFILE) ./...
	@$(GO) tool cover -func=$(COVERPROFILE) | tail -1
	@echo
	@echo "Codecov applies the ignore: rules in codecov.yml (examples/, generated"
	@echo "protobuf), so the reported total reads roughly 13 points above this one."
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
build-matrix: ## Compile every supported cgo tag combination
	@echo "==> cgo-free (the default build)"
	CGO_ENABLED=0 $(GO) build ./...
	@echo "==> cgo, no tags"
	CGO_ENABLED=1 $(GO) build ./...
	@echo "==> -tags libsoxr"
	CGO_ENABLED=1 $(GO) build -tags libsoxr ./...
	@echo "==> -tags libopus"
	CGO_ENABLED=1 $(GO) build -tags libopus ./...
	@echo "==> -tags 'libsoxr libopus'"
	CGO_ENABLED=1 $(GO) build -tags "libsoxr libopus" ./...

##@ Generate

.PHONY: tools-proto
tools-proto: ## Install buf and the protoc plugins at their pinned versions
	$(GO) install github.com/bufbuild/buf/cmd/buf@$(BUF_VERSION)
	$(GO) install google.golang.org/protobuf/cmd/protoc-gen-go@$(PROTOC_GEN_GO_VERSION)
	$(GO) install google.golang.org/grpc/cmd/protoc-gen-go-grpc@$(PROTOC_GEN_GO_GRPC_VERSION)

.PHONY: generate
generate: tools-proto ## Regenerate the Riva protobuf clients from the .proto files
	$(GO) generate ./...

.PHONY: generate-check
generate-check: generate ## Fail if the checked-in generated code is stale
	@if ! git diff --quiet -- provider/nvidia/internal/rivapb; then \
		echo "generated protobuf is out of date, commit the result of: make generate" >&2; \
		git --no-pager diff --stat -- provider/nvidia/internal/rivapb >&2; \
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

##@ Docs

.PHONY: require-hugo
require-hugo:
	@command -v hugo >/dev/null || { \
		echo "hugo (extended) $(HUGO_VERSION) is required: https://gohugo.io/installation/" >&2; \
		exit 1; \
	}

.PHONY: docs-serve
docs-serve: require-hugo ## Serve the documentation site with live reload
	cd $(WEBSITE_DIR) && hugo serve --buildDrafts --printPathWarnings

.PHONY: docs-build
docs-build: require-hugo ## Build the documentation site into website/public
	cd $(WEBSITE_DIR) && hugo --gc --minify --printPathWarnings \
		$(if $(HUGO_BASEURL),--baseURL "$(HUGO_BASEURL)")

.PHONY: docs-check
docs-check: require-hugo ## Fail on any unresolved link in the documentation
	@cd $(WEBSITE_DIR) && hugo --gc --renderToMemory --logLevel warn 2>&1 | tee $(HUGO_CHECK_LOG)
	@# The ::error:: prefix is a GitHub Actions annotation, omitted off CI.
	@if grep -F "render-link: unresolved" $(HUGO_CHECK_LOG); then \
		echo "$${GITHUB_ACTIONS:+::error::}unresolved documentation links, see the warnings above" >&2; \
		exit 1; \
	fi

##@ Native dependencies

# These three follow docker/build.Dockerfile and docker/runtime.Dockerfile,
# which are Debian and Linux only, so the host install is too. On another system
# install the same four libraries by hand: libsoxr, libopus, the ONNX Runtime
# and RNNoise.
.PHONY: require-linux
require-linux:
	@[[ "$$(uname -s)" == "Linux" ]] || { \
		echo "this target is Linux only, see the comment above it in the Makefile" >&2; \
		exit 1; \
	}

.PHONY: deps
deps: require-linux ## Install the cgo headers the libsoxr and libopus tags link against
	sudo apt-get update
	sudo apt-get install -y libsoxr-dev libopus-dev pkg-config

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
