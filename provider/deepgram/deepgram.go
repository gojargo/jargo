// Package deepgram provides Deepgram's streaming speech-to-text service (over
// the live transcription WebSocket, see stt.go), its Aura text-to-speech
// service (see tts.go), and its Flux streaming STT and TTS services (see
// flux.go and flux_tts.go).
//
// The STT service pushes InterimTranscriptionFrames and TranscriptionFrames
// downstream. It asks Deepgram to flush the transcript as soon as the speech
// ends rather than wait on Deepgram's own endpointing, and the answer to that
// request is what marks the end of the user's turn.
package deepgram

import (
	"context"
	"errors"
	"log/slog"
	"strings"
	"sync"
	"time"

	"github.com/coder/websocket"
	"github.com/gojargo/jargo/service/wsutil"
)

// authToken formats the shared Deepgram Authorization header value.
func authToken(apiKey string) string { return "Token " + apiKey }

// closeStreamMessage tells a transcription session no more audio is coming.
const closeStreamMessage = `{"type":"CloseStream"}`

// sendCloseStream tells a transcription session on conn that no more audio is
// coming, and reports whether the socket is still open for the closing
// handshake. writeMu is the session's own write lock.
//
// The message is bounded by wait (wsutil.DefaultCloseTimeout when not
// positive). An open session holds one of the account's concurrent connections,
// and a send stuck behind a peer that has stopped reading would otherwise hold
// the teardown, and the connection with it, for as long as the peer stays
// silent: once wait passes the socket is dropped instead, which also releases
// the stuck send.
func sendCloseStream(ctx context.Context, conn *wsutil.Conn, writeMu *sync.Mutex, wait time.Duration) bool {
	if wait <= 0 {
		wait = wsutil.DefaultCloseTimeout
	}
	// The session's own context may already have ended, and the message is
	// what the teardown is waiting on, so it goes out on one of its own.
	ctx, cancel := context.WithTimeout(context.WithoutCancel(ctx), wait)
	defer cancel()
	done := make(chan struct{})
	go func() {
		defer close(done)
		writeMu.Lock()
		defer writeMu.Unlock()
		if ctx.Err() != nil {
			return
		}
		_ = conn.Write(ctx, websocket.MessageText, []byte(closeStreamMessage))
	}()
	select {
	case <-done:
		return true
	case <-ctx.Done():
		slog.Debug("deepgram did not take the close in time, dropping the connection", "timeout", wait)
		_ = conn.CloseNow()
		return false
	}
}

// defaultHost is the Deepgram API host the services address when no base URL is
// configured.
const defaultHost = "api.deepgram.com"

// deriveDeepgramURLs derives paired WebSocket and HTTP URLs from a single base
// URL.
//
// The streaming and REST endpoints have to agree on how secure the connection
// is, so a caller names the host once and both are derived from it, preserving
// the security level chosen. That matters for air-gapped or private
// deployments, where the insecure schemes are acceptable.
//
// Accepted inputs:
//   - "wss://" or "https://": secure, paired as wss + https
//   - "ws://" or "http://": insecure, paired as ws + http
//   - a bare hostname with no scheme: secure
//   - an unrecognized scheme: warns, and secure
//
// The host keeps whatever port and path came with it.
func deriveDeepgramURLs(baseURL string) (wsURL, httpURL string) {
	var scheme, host string
	if s, h, found := strings.Cut(baseURL, "://"); found {
		scheme, host = s+"://", h
		switch scheme {
		case "wss://", "https://", "ws://", "http://":
		default:
			slog.Warn("unrecognized scheme in the Deepgram base URL, defaulting to wss:// and https://",
				"base_url", baseURL)
		}
	} else {
		host = baseURL
	}

	if scheme == "ws://" || scheme == "http://" {
		return "ws://" + host, "http://" + host
	}
	return "wss://" + host, "https://" + host
}

// The following symbols are shared by the Flux STT and TTS streams (flux.go and
// flux_tts.go); each modality file holds only its own logic.
const (
	// defaultTTSSampleRate is the default PCM rate for Deepgram's TTS services
	// (Aura and Flux TTS).
	defaultTTSSampleRate = 24000
	// fluxEncoding is the sole audio encoding the Flux STT and TTS streams use:
	// signed little-endian 16-bit PCM.
	fluxEncoding = "linear16"
	// fluxMsgError is the Flux WebSocket message type carrying a fatal
	// server-side error; both the Flux STT and TTS streams handle it.
	fluxMsgError = "Error"
)

// errFluxServer is returned when a Flux stream reports a fatal server-side error.
// It is shared by the Flux STT and TTS streams.
//
//nolint:gochecknoglobals // sentinel error
var errFluxServer = errors.New("deepgram flux: server error")
