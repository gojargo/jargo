// Package zoomaiutil holds the helpers shared by everything that calls Zoom AI
// Services: the tracking ID and headers every request carries, and the check
// on the endpoint a request is sent to.
package zoomaiutil

import (
	"log/slog"
	"net/http"
	"net/url"
	"slices"
	"strings"

	"github.com/google/uuid"
)

const (
	// HeaderTrackingID carries the ID Zoom support uses to find a request or
	// session.
	HeaderTrackingID = "x-zm-trackingid"

	// userAgent identifies the client on every request.
	userAgent = "jargo"
	// trackingIDPrefix opens every tracking ID this client sends.
	trackingIDPrefix = "zoom_ai_community_jargo_"
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

// NewTrackingID returns a new x-zm-trackingid, which Zoom support uses to find
// a request or session.
func NewTrackingID() string {
	return trackingIDPrefix + uuid.NewString()
}

// SetRequestHeaders sets the headers every Zoom AI Services request carries:
// the API key (or JWT) as a bearer token, the client, and the request's
// tracking ID.
func SetRequestHeaders(header http.Header, apiKey, trackingID string) {
	header.Set("Authorization", "Bearer "+apiKey)
	header.Set("User-Agent", userAgent)
	header.Set(HeaderTrackingID, trackingID)
}

// CheckURL returns endpoint, warning if it sends the API key unencrypted to a
// remote host. service names the service and data what it sends besides the
// API key, for the warning.
func CheckURL(endpoint, service, data string) string {
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
