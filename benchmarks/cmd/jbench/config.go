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
// this is, what it ran on (down to the CPUs it was allowed, their frequency
// governor, the kernel and the cgroup limits), and which scenario. The Python side prints the same
// keys, read the same way, so benchstat lines the two up; only framework,
// framework_version and runtime differ between them.
func configLines(scenarioName, version string) [][2]string {
	allowed := cpusAllowed()
	return [][2]string{
		{"framework", "jargo"},
		{"framework_version", version},
		{"runtime", runtime.Version()},
		{"goos", runtime.GOOS},
		{"goarch", runtime.GOARCH},
		{"cpu", cpuModel()},
		{"cpus", strconv.Itoa(runtime.NumCPU())},
		{"cpus_allowed", allowed},
		{"governor", governor(allowed)},
		{"kernel", readValue("/proc/sys/kernel/osrelease")},
		{"cgroup", cgroupVersion()},
		{"memory_limit", readValue("/sys/fs/cgroup/memory.max")},
		{"scenario", scenarioName},
	}
}

// cpusAllowed is the list of CPUs the process may run on, as the kernel
// writes it ("2-3", "0,2"), from /proc/self/status.
func cpusAllowed() string {
	data, err := os.ReadFile("/proc/self/status")
	if err != nil {
		return unknown
	}
	for line := range strings.Lines(string(data)) {
		if rest, ok := strings.CutPrefix(line, "Cpus_allowed_list:"); ok {
			return strings.TrimSpace(rest)
		}
	}
	return unknown
}

// governor is the frequency governor of the first CPU the process may run on.
// A container usually still sees the host's, through a read-only /sys.
func governor(allowed string) string {
	end := strings.IndexFunc(allowed, func(r rune) bool { return r < '0' || r > '9' })
	if end < 0 {
		end = len(allowed)
	}
	if end == 0 {
		return unknown
	}
	return readValue("/sys/devices/system/cpu/cpu" + allowed[:end] + "/cpufreq/scaling_governor")
}

// cgroupVersion is v2 where the unified hierarchy is mounted, v1 otherwise.
func cgroupVersion() string {
	if _, err := os.Stat("/sys/fs/cgroup/cgroup.controllers"); err == nil {
		return "v2"
	}
	return "v1"
}

// readValue is the trimmed content of a one-line file, or "unknown".
func readValue(path string) string {
	data, err := os.ReadFile(path) //nolint:gosec // fixed paths under /proc and /sys
	if err != nil {
		return unknown
	}
	if v := strings.TrimSpace(string(data)); v != "" {
		return v
	}
	return unknown
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
