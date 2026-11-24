// Package realtime is a speech-to-speech service built on Rime's Prism
// conversations. A single WebSocket carries the conversation: input audio
// streams up, and Prism streams its spoken reply, the text of it, the
// transcript of what the user said and its own turn detection back down.
//
// Place the service where the STT/LLM/TTS stack would go, between the user and
// the assistant aggregators of one shared conversation. The first conversation
// frame opens the Prism conversation with the conversation's messages and
// starts a reply. Messages of an LLMMessagesAppendFrame are added to Prism's
// history as they arrive, and with RunLLM set Prism replies when the assistant
// aggregator asks for a response, unless a reply that started after the write
// already took them in. The service only appends to Prism's history: replacing
// or transforming the conversation's messages mid-conversation does not change
// it, and an LLMRunFrame or a replacement asking to run after the opening
// starts no reply. Use an LLMMessagesAppendFrame with RunLLM set to ask for
// one. A write Prism refuses does not roll the local conversation back.
//
// Tool results reach Prism as they are produced. An intermediate result is a
// history message beside the call, which stays open until its final result is
// submitted. Prism answers results when the assistant aggregator asks for a
// response, continuing the tool round when it is still current. Prism mentions
// the result of an asynchronous call that arrives after its round closed only
// when asked, so the service asks for a reply that reports it.
//
// Playback positions are estimates, and no completion receipts are sent unless
// the audio is discarded. Speech hints do not change the output volume.
//
// Use one service per conversation. Nothing reconnects or replays on its own:
// after an operation whose outcome is unknown or partial, start a fresh service
// with a history that is known to be valid and complete.
//
// Prism sends 16-bit mono PCM at 24 kHz, so run the output transport at that
// rate. Input audio is taken at any rate from 8 to 192 kHz.
package realtime

import (
	"time"

	"github.com/gojargo/jargo/internal/validate"
)

// The events the service raises, beside every processor's.
const (
	// EventSessionUpdated fires with the session information Prism
	// acknowledged, when the conversation opens and after each settings change.
	//
	//	events.On(svc.Events(), realtime.EventSessionUpdated,
	//	    func(ctx context.Context, info realtime.SessionInfo) { … })
	EventSessionUpdated = "on_session_updated"
	// EventRimeEvent fires with every event of the Prism session, provider
	// diagnostics included.
	//
	//	events.On(svc.Events(), realtime.EventRimeEvent,
	//	    func(ctx context.Context, event *realtime.Event) { … })
	EventRimeEvent = "on_rime_event"
)

// The playback modes.
const (
	// PlaybackEstimated plays the audio and estimates how much of a reply was
	// heard when it is interrupted. It is the default.
	PlaybackEstimated = "estimated"
	// PlaybackDiscard discards all audio, and reports every reply as played
	// for no time at all.
	PlaybackDiscard = "discard"
)

const (
	// maxPendingOperations bounds the operations and the requests waiting. The
	// session refuses more than 128 pending operations of its own.
	maxPendingOperations = 128
	// maxInputAudioBytes bounds the audio waiting to be sent: 20 s of 48 kHz
	// mono PCM.
	maxInputAudioBytes = 1_920_000
	// defaultShutdownTimeout bounds the drain of a graceful stop.
	defaultShutdownTimeout = 5 * time.Second
	// maxRetainedText bounds the text kept of one transcript or one message.
	maxRetainedText = 65536
	// maxRetainedCalls bounds the tool calls kept.
	maxRetainedCalls = 1024
	// toolReportInstruction asks for a reply that reports a tool result.
	toolReportInstruction = "Continue according to the agent's instructions using this tool result. " +
		"Reply or call a tool as appropriate."
)

// Config configures the Prism service.
type Config struct {
	// APIKey is the Rime API key. Required.
	APIKey string `validate:"required"`
	// Endpoint overrides the Prism WebSocket endpoint with a full
	// ws:// or wss:// URL ending in /v1/realtime. ws:// is taken for a local
	// development server only.
	Endpoint string
	// Settings are the initial settings and the tools advertised. A field left
	// not given takes its default.
	Settings Settings
	// TurnDetection holds the server's turn settings for the opening, such as
	// the eagerness and the least silence that ends a turn. Its interruption
	// rules may be given in Settings instead, which can also change them
	// later, but not in both.
	TurnDetection *TurnDetection `validate:"omitempty"`
	// Timeouts bound the connection, the readiness and the acknowledgments.
	Timeouts Timeouts
	// RequestID is the correlation id asked of the provider: nonblank printable
	// ASCII. The session information reports the one Prism actually used.
	RequestID string
	// Startup bounds the opening of the conversation and its retries.
	Startup StartupParams
	// Playback is "estimated" for normal output or "discard" to discard all
	// audio and report every reply as unplayed; empty uses "estimated".
	Playback string `validate:"omitempty,oneof=estimated discard"`
}

// Validate reports whether the configuration is usable.
func (c Config) Validate() error {
	if err := validate.Struct(c); err != nil {
		return err
	}
	_, err := newSettings(c)
	return err
}
