# Security Policy

## Supported versions

jargo is in early development and its version stays in the `0.0.x` range. Only
the latest release receives security fixes; there are no long-term-support
branches yet. The supported surface will be revisited once `0.1.0` ships.

| Version | Supported          |
| ------- | ------------------ |
| latest `0.0.x` | :white_check_mark: |
| older   | :x:                |

## Reporting a vulnerability

Please report security vulnerabilities **privately** — do not open a public
issue, pull request, or discussion for them.

Use GitHub's private vulnerability reporting:

1. Go to the repository's **Security** tab.
2. Click **Report a vulnerability**
   ([direct link](https://github.com/gojargo/jargo/security/advisories/new)).
3. Describe the issue, including affected versions and, where possible, a
   minimal reproduction.

We aim to acknowledge a report within a few business days and will keep you
updated as we investigate and prepare a fix. Once a fix is available we will
coordinate disclosure and credit you in the advisory, unless you prefer to
remain anonymous.

## Verifying a release

Releases are signed, and the release workflow also attests build provenance for
what it builds. Both are worth checking before you run a downloaded binary.
Releases up to and including `v0.1.0` predate the provenance step and carry
signatures only.

**Checksums and signature.** GoReleaser signs `checksums.txt` with cosign
keyless, which transitively covers every artifact the file lists. Download the
archive, `checksums.txt`, `checksums.txt.sig` and `checksums.txt.pem` from the
release, then:

```sh
cosign verify-blob \
  --certificate checksums.txt.pem \
  --signature checksums.txt.sig \
  --certificate-identity-regexp '^https://github.com/gojargo/jargo/' \
  --certificate-oidc-issuer https://token.actions.githubusercontent.com \
  checksums.txt
sha256sum --check --ignore-missing checksums.txt
```

The identity flags are what make the check meaningful: without them cosign
confirms only that somebody signed the file, not that this project's release
workflow did.

**Build provenance.** The release workflow attests the artifacts with GitHub
artifact attestations, which record the workflow and the commit that produced
them. Verify with the GitHub CLI:

```sh
gh attestation verify jargo_<version>_linux_amd64.tar.gz --repo gojargo/jargo
```

Attestations are held in GitHub's attestation store rather than attached to the
release, so there is no file to download for this step.

## Scope

jargo is a library and a set of example bots. Vulnerabilities in jargo's own
code — the pipeline, transports, providers, and audio handling — are in scope.
Issues in upstream dependencies, the underlying ONNX Runtime, or third-party
provider APIs should be reported to those projects, though we welcome a heads-up
if jargo's use of them is affected.
