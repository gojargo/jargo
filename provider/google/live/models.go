package live

import (
	"regexp"
	"strconv"
	"strings"

	"github.com/gojargo/jargo/provider/google/gemini"
)

// modelVersion captures the version a Live model id carries. An id names it with
// or without a minor part ("gemini-3.8-live", "gemini-2.0-flash-live-001").
//
//nolint:gochecknoglobals // compiled once, read-only
var modelVersion = regexp.MustCompile(`gemini(?:-[a-z]+)*-(\d+)(?:\.(\d+))?`)

// version is the major and minor version of a model id, and whether it carries
// one at all. A model whose version cannot be read is not assumed to be any
// particular generation: each capability below says what it does with that.
func version(model string) (major, minor int, known bool) {
	m := modelVersion.FindStringSubmatch(model)
	if m == nil {
		return 0, 0, false
	}
	major, err := strconv.Atoi(m[1])
	if err != nil {
		return 0, 0, false
	}
	if m[2] != "" {
		minor, err = strconv.Atoi(m[2])
		if err != nil {
			return 0, 0, false
		}
	}
	return major, minor, true
}

// atLeast reports whether the model is of generation wantMajor.wantMinor or
// later. A model whose version cannot be read is not.
func atLeast(model string, wantMajor, wantMinor int) bool {
	major, minor, known := version(model)
	if !known {
		return false
	}
	return major > wantMajor || (major == wantMajor && minor >= wantMinor)
}

// expectsInteractionStatus reports whether the model reports its interaction
// status. The thinking models from 3.8 on reason in the background between
// chunks of output, so a completed generation does not mean the turn is over.
// The version gate leaves out the earlier models named for thinking, which do
// not report the status.
func expectsInteractionStatus(model string) bool {
	return strings.Contains(model, "thinking") && atLeast(model, 3, 8)
}

// supportsNonBlockingTools reports whether the model takes a function
// declaration that says whether to wait for the call. The 3.x models before 3.8
// do not; the thinking models require it. A model of no readable generation is
// assumed to take it, as is anything before 3.
func supportsNonBlockingTools(model string) bool {
	major, _, known := version(model)
	return !known || major < 3 || atLeast(model, 3, 8)
}

// toolsDefaultToNonBlocking reports whether the model runs a function call
// without waiting for it unless the declaration asks it to. The 3.8 family
// flipped that default.
func toolsDefaultToNonBlocking(model string) bool { return atLeast(model, 3, 8) }

// supportsBlockingTools reports whether the model accepts a declaration asking
// it to wait. The thinking models run every call without waiting and refuse one.
func supportsBlockingTools(model string) bool { return !expectsInteractionStatus(model) }

// lowestThinkingLevel is the least the Live thinking models will think. They
// reject "minimal", and the lowest they take keeps the reply latency down, which
// matches what the streaming service defaults its own flash models to.
const lowestThinkingLevel = "LOW"

// resolvedThinking is the thinking configuration the session is opened with.
//
// The Live thinking models require a level: the API has no default for them and
// refuses a setup that sets none. An unset level therefore defaults to the
// lowest they accept, and a configured one is never overridden. The rest of the
// configuration is carried through, and the caller's own value is left as it
// stands.
func resolvedThinking(model string, cfg *gemini.ThinkingConfig) *gemini.ThinkingConfig {
	if !expectsInteractionStatus(model) {
		return cfg
	}
	if cfg != nil && cfg.Level != "" {
		return cfg
	}
	resolved := gemini.ThinkingConfig{Level: lowestThinkingLevel}
	if cfg != nil {
		resolved = *cfg
		resolved.Level = lowestThinkingLevel
	}
	return &resolved
}
