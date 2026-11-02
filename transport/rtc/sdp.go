package rtc

import (
	"log/slog"
	"regexp"
	"strings"
)

// splitLines splits text into lines on any line ending, dropping the endings.
func splitLines(text string) []string {
	text = strings.ReplaceAll(text, "\r\n", "\n")
	text = strings.ReplaceAll(text, "\r", "\n")
	lines := strings.Split(text, "\n")
	if n := len(lines); n > 0 && lines[n-1] == "" {
		lines = lines[:n-1]
	}
	return lines
}

// sdpCleanupICECandidates keeps only the candidates matching pattern that are
// not relayed or reflexive (they carry no raddr).
func sdpCleanupICECandidates(text, pattern string) string {
	slog.Debug("rtc: removing unsupported ICE candidates from SDP")
	re, err := regexp.Compile(pattern)
	match := func(line string) bool {
		if err != nil {
			return strings.Contains(line, pattern)
		}
		return re.MatchString(line)
	}
	var result []string
	for _, line := range splitLines(text) {
		if strings.Contains(line, "a=candidate") {
			if match(line) && !strings.Contains(line, "raddr") {
				result = append(result, line)
			}
		} else {
			result = append(result, line)
		}
	}
	return strings.Join(result, "\r\n") + "\r\n"
}

// sdpCleanupFingerprints removes the sha-384 and sha-512 fingerprints, which
// some clients cannot handle.
func sdpCleanupFingerprints(text string) string {
	slog.Debug("rtc: removing unsupported fingerprints from SDP")
	var result []string
	for _, line := range splitLines(text) {
		if !strings.Contains(line, "sha-384") && !strings.Contains(line, "sha-512") {
			result = append(result, line)
		}
	}
	return strings.Join(result, "\r\n") + "\r\n"
}

// SDPMunging rewrites an answer for clients with a minimal WebRTC stack, such as
// an ESP32: it drops the fingerprint algorithms they cannot handle and, when
// host is set, every ICE candidate except the direct ones for that host.
func SDPMunging(sdp, host string) string {
	sdp = sdpCleanupFingerprints(sdp)
	if host != "" {
		sdp = sdpCleanupICECandidates(sdp, host)
	}
	return sdp
}
