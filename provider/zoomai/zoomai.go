// Package zoomai provides Zoom AI Services text-to-speech: an HTTP service
// that sends each sentence in its own request and streams the audio back.
//
// See https://developers.zoom.us/docs/ai-services/tts/ and https://www.zoom.ai/.
package zoomai

import (
	"log/slog"
	"net/http"
	"net/url"
	"slices"
	"strings"

	"github.com/google/uuid"
)

const (
	// userAgent identifies the client on every request.
	userAgent = "jargo"
	// trackingIDPrefix opens every tracking ID this client sends.
	trackingIDPrefix = "zoom_ai_community_jargo_"
	// headerTrackingID carries the ID Zoom support uses to find a request or
	// session.
	headerTrackingID = "x-zm-trackingid"
)

// plaintextSchemes are the URL schemes that send a request unencrypted.
//
//nolint:gochecknoglobals // read-only lookup table
var plaintextSchemes = []string{"http", "ws"}

// loopbackHosts are the hosts a plaintext request does not leave the machine
// for.
//
//nolint:gochecknoglobals // read-only lookup table
var loopbackHosts = []string{"localhost", "127.0.0.1", "::1"}

// newTrackingID returns a new x-zm-trackingid, which Zoom support uses to find
// a request or session.
func newTrackingID() string {
	return trackingIDPrefix + uuid.NewString()
}

// setRequestHeaders sets the headers every Zoom AI Services request carries:
// the API key (or JWT) as a bearer token, the client, and the request's
// tracking ID.
func setRequestHeaders(header http.Header, apiKey, trackingID string) {
	header.Set("Authorization", "Bearer "+apiKey)
	header.Set("User-Agent", userAgent)
	header.Set(headerTrackingID, trackingID)
}

// checkURL returns endpoint, warning if it sends the API key unencrypted to a
// remote host. service names the service and data what it sends besides the
// API key, for the warning.
func checkURL(endpoint, service, data string) string {
	parsed, err := url.Parse(endpoint)
	if err != nil {
		return endpoint
	}
	scheme := strings.ToLower(parsed.Scheme)
	if slices.Contains(plaintextSchemes, scheme) && !slices.Contains(loopbackHosts, parsed.Hostname()) {
		slog.Warn(service+" url is not encrypted; the API key and "+data+" are sent in plaintext",
			"scheme", scheme)
	}
	return endpoint
}
