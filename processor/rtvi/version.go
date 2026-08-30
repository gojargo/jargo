package rtvi

import (
	"runtime/debug"
	"strconv"
	"strings"
	"sync"
)

// modulePath is this library's module path, which is how its version is found
// among the build information of whatever program embeds it.
const modulePath = "github.com/gojargo/jargo"

//nolint:gochecknoglobals // read once, from build information that cannot change
var libraryVersion = sync.OnceValue(func() string {
	info, ok := debug.ReadBuildInfo()
	if !ok {
		return ""
	}
	if info.Main.Path == modulePath {
		return info.Main.Version
	}
	for _, dep := range info.Deps {
		if dep.Path == modulePath {
			return dep.Version
		}
	}
	return ""
})

// LibraryVersion is the version of this library, as recorded in the build
// information of the program running it. It is empty for a program built
// without it, a binary built from a local checkout being the usual case.
func LibraryVersion() string { return libraryVersion() }

// parseProtocolVersion reads a protocol version of the form "major.minor.patch"
// into its three numbers. Anything else is refused: the negotiation turns on the
// major version, so a version that cannot be read is not one to guess at.
func parseProtocolVersion(v string) ([3]int, bool) {
	var out [3]int
	parts := strings.Split(v, ".")
	if len(parts) != len(out) {
		return out, false
	}
	for i, part := range parts {
		n, err := strconv.Atoi(part)
		if err != nil {
			return [3]int{}, false
		}
		out[i] = n
	}
	return out, true
}

// formatProtocolVersion writes a parsed version back out.
func formatProtocolVersion(v [3]int) string {
	return strconv.Itoa(v[0]) + "." + strconv.Itoa(v[1]) + "." + strconv.Itoa(v[2])
}

// protocolMajor is the major version of the protocol this implementation
// speaks, which is the half of the version compatibility turns on.
func protocolMajor() int {
	v, _ := parseProtocolVersion(ProtocolVersion)
	return v[0]
}
