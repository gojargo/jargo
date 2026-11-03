package main

import (
	"os"
	"runtime"
	"runtime/debug"
	"strconv"
	"strings"
)

// unknown stands for a value that could not be read.
const unknown = "unknown"

// configLines are the benchfmt config lines that open the output: which side
// this is, what it ran on, and which scenario. The Python side prints the same
// keys, read the same way, so benchstat lines the two up; only framework,
// framework_version and runtime differ between them.
func configLines(scenarioName, version string) [][2]string {
	return [][2]string{
		{"framework", "jargo"},
		{"framework_version", version},
		{"runtime", runtime.Version()},
		{"goos", runtime.GOOS},
		{"goarch", runtime.GOARCH},
		{"cpu", cpuModel()},
		{"cpus", strconv.Itoa(runtime.NumCPU())},
		{"scenario", scenarioName},
	}
}

// buildVersion is the revision the binary was built from: the linker-set one
// if there is one, else the VCS stamp of the build, marked when the tree had
// changes.
func buildVersion() string {
	if frameworkVersion != "" {
		return frameworkVersion
	}
	info, ok := debug.ReadBuildInfo()
	if !ok {
		return unknown
	}
	var revision string
	var modified bool
	for _, s := range info.Settings {
		switch s.Key {
		case "vcs.revision":
			revision = s.Value
		case "vcs.modified":
			modified = s.Value == "true"
		}
	}
	if revision == "" {
		return unknown
	}
	if len(revision) > 12 {
		revision = revision[:12]
	}
	if modified {
		revision += "-dirty"
	}
	return revision
}

// cpuModel is the processor's model name from /proc/cpuinfo, or "unknown"
// where there is none.
func cpuModel() string {
	data, err := os.ReadFile("/proc/cpuinfo")
	if err != nil {
		return unknown
	}
	for line := range strings.Lines(string(data)) {
		key, value, ok := strings.Cut(line, ":")
		if ok && strings.TrimSpace(key) == "model name" {
			return strings.TrimSpace(value)
		}
	}
	return unknown
}
