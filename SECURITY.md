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

## Scope

jargo is a library and a set of example bots. Vulnerabilities in jargo's own
code — the pipeline, transports, providers, and audio handling — are in scope.
Issues in upstream dependencies, the underlying ONNX Runtime, or third-party
provider APIs should be reported to those projects, though we welcome a heads-up
if jargo's use of them is affected.
