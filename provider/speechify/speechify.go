// Package speechify provides Speechify text-to-speech over the HTTP API. Each
// sentence is one request to the /v1/audio/stream/with-timestamps endpoint,
// which streams the PCM audio and word-level speech marks back as server-sent
// events.
package speechify

import (
	"runtime/debug"
	"sync"
)

const (
	// defaultURL is the Speechify API base.
	defaultURL = "https://api.speechify.ai"
	// streamPath is the endpoint that streams audio and speech marks.
	streamPath = "/v1/audio/stream/with-timestamps"
	// apiVersion is pinned so behavior doesn't depend on the workspace
	// default; pcm_16000 is 16 kHz from this version on.
	apiVersion = "2026-09-30"
	// defaultSampleRate is the rate synthesized when the configured one is not
	// a rate Speechify renders.
	defaultSampleRate = 24000
	defaultModel      = "simba-3.5"
	defaultVoice      = "geffen_32"
	// callerName identifies the integration to Speechify's platform
	// attribution, as the Speechify-Caller header.
	callerName = "jargo"
	// modulePath is this library's module path, which is how its version is
	// found among the build information of the program embedding it.
	modulePath = "github.com/gojargo/jargo"
)

// pcmSampleRates are the rates Speechify synthesizes, as the pcm_<rate> output
// formats.
//
//nolint:gochecknoglobals // read-only lookup table
var pcmSampleRates = []int{8000, 16000, 22050, 24000, 44100, 48000}

// callerVersion is this library's version, sent as the
// Speechify-Caller-Version header. It is empty for a program built without
// it, a binary built from a local checkout being the usual case.
//
//nolint:gochecknoglobals // read once, from build information that cannot change
var callerVersion = sync.OnceValue(func() string {
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
