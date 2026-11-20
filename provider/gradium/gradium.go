// Package gradium provides Gradium's speech services: streaming text-to-speech
// and real-time streaming speech-to-text, both over Gradium's WebSocket API.
// The TTS side holds one connection open for the call and speaks each turn as a
// request of its own on it, streaming the raw PCM audio downstream with the
// time each word is spoken; the STT side streams 80 ms audio chunks
// and surfaces recognized text as interim and finalized transcriptions.
package gradium

import (
	"context"
	"log/slog"
	"strings"
)

// Wire-protocol message keys and values shared by the STT and TTS transports.
const (
	// msgError is the server message type reporting an error.
	msgError = "error"
	// msgLog is the server message type reporting a problem with the session's
	// setup, sent only to a session that asks for them with a log level.
	msgLog = "log"
	// msgType is the message discriminator key.
	msgType = "type"
	// msgAudio is the base64 PCM audio field key.
	msgAudio = "audio"
	// msgText is the transcript/text field key.
	msgText = "text"
	// msgFlushed says the buffered audio has been processed.
	msgFlushed = "flushed"
	// msgEndStream is the end-of-stream message type.
	msgEndStream = "end_of_stream"
	// keyClientReqID is the client request-id field key.
	keyClientReqID = "client_req_id"
	// keyLogLevel is the setup field asking for the server's log messages.
	keyLogLevel = "log_level"
	// encPCM is the raw PCM encoding value.
	encPCM = "pcm"
)

// LogLevel is the lowest level of the server's log messages a session asks
// for. The server reports problems with the session's setup this way, such as
// an unknown or deprecated json_config key, and the session carries on.
type LogLevel string

// The levels a session can ask for the server's log messages at.
const (
	LogLevelDebug   LogLevel = "debug"
	LogLevelInfo    LogLevel = "info"
	LogLevelWarning LogLevel = "warning"
	LogLevelError   LogLevel = "error"
)

// logServerMessage logs a server log message at the level the server gave it.
// A level this does not know is logged as a warning.
func logServerMessage(ctx context.Context, level, message string) {
	name := strings.ToLower(level)
	var l slog.Level
	switch LogLevel(name) {
	case LogLevelDebug:
		l = slog.LevelDebug
	case LogLevelInfo:
		l = slog.LevelInfo
	case LogLevelError:
		l = slog.LevelError
	case LogLevelWarning:
		l = slog.LevelWarn
	default:
		name, l = string(LogLevelWarning), slog.LevelWarn
	}
	slog.Log(ctx, l, "gradium "+name+": "+message)
}
