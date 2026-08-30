// Package security holds the checks that decide whether a connection from
// outside is allowed in.
package security

import "strings"

// IsOriginAllowed reports whether origin, the value of a request's Origin
// header, is permitted by allowed.
//
// An empty allowed list permits every origin, including a request that carries
// no Origin header at all: that is the default, and it is what a telephony
// provider streaming a call needs, since it is not a browser and sends no
// origin. A non-empty list permits only the origins it names, matched whole and
// without regard to case, and rejects a request whose origin is missing.
//
// It is the guard against a page on another site opening a WebSocket to this
// one on a visitor's behalf: the browser sends the page's origin, and a server
// that never looks at it accepts the connection as readily as its own client's.
func IsOriginAllowed(origin string, allowed []string) bool {
	if len(allowed) == 0 {
		return true
	}
	for _, a := range allowed {
		if strings.EqualFold(origin, a) {
			return true
		}
	}
	return false
}
