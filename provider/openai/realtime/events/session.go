// Package events holds the messages of OpenAI's Realtime API: the session's
// properties, the conversation items, and the events the client sends and the
// server sends back.
package events

import (
	"encoding/json"
	"errors"
	"fmt"
	"slices"

	"github.com/gojargo/jargo/frames"
)

// SampleRate is the fixed rate of the Realtime API's PCM audio.
const SampleRate = 24000

// WhisperModel is the transcription model a transcription configured without
// one uses.
const WhisperModel = "gpt-realtime-whisper"

// Modality is something the model can respond with.
type Modality string

// The modalities.
const (
	ModalityText  Modality = "text"
	ModalityAudio Modality = "audio"
)

// AudioFormat is the format of input or output audio.
type AudioFormat struct {
	// Type is "audio/pcm", "audio/pcmu" or "audio/pcma".
	Type string `json:"type"`
	// Rate is the sample rate, which PCM fixes at 24000.
	Rate int `json:"rate,omitempty"`
}

// PCM is 16-bit PCM at 24 kHz.
func PCM() *AudioFormat { return &AudioFormat{Type: "audio/pcm", Rate: SampleRate} }

// PCMU is G.711 mu-law.
func PCMU() *AudioFormat { return &AudioFormat{Type: "audio/pcmu"} }

// PCMA is G.711 A-law.
func PCMA() *AudioFormat { return &AudioFormat{Type: "audio/pcma"} }

// InputAudioTranscription configures the transcription of the user's audio.
type InputAudioTranscription struct {
	// Model is the transcription model (e.g. "gpt-realtime-whisper",
	// "gpt-4o-transcribe", "whisper-1"); empty uses WhisperModel.
	Model string `json:"model"`
	// Language is the language to transcribe, as a code.
	Language string `json:"language,omitempty"`
	// Prompt is text that guides the transcription.
	Prompt string `json:"prompt,omitempty"`
	// Delay is how long the model waits before emitting transcription text, one
	// of "minimal", "low", "medium", "high" or "xhigh". Higher values trade
	// latency for accuracy. Supported by "gpt-realtime-whisper".
	Delay string `json:"delay,omitempty"`
	// Keywords are words or phrases that guide transcription of the input
	// audio. Supported by "gpt-transcribe" and "gpt-live-transcribe".
	Keywords []string `json:"keywords,omitempty"`
	// Languages are the possible languages of the input audio, as ISO-639-1
	// codes. Supported by "gpt-transcribe" and "gpt-live-transcribe".
	Languages []string `json:"languages,omitempty"`
}

// MarshalJSON implements json.Marshaler, sending WhisperModel for an empty
// model.
func (t InputAudioTranscription) MarshalJSON() ([]byte, error) {
	type plain InputAudioTranscription
	if t.Model == "" {
		t.Model = WhisperModel
	}
	return json.Marshal(plain(t))
}

// Turn detection types.
const (
	TurnDetectionServerVAD   = "server_vad"
	TurnDetectionSemanticVAD = "semantic_vad"
)

// TurnDetection configures how the server detects the user's turns: by voice
// activity, or by what the user said. A nil TurnDetection leaves the server's
// default, which detects them by voice activity; one with Disabled set turns the
// server's detection off, leaving the pipeline's own turn detection to drive the
// session.
type TurnDetection struct {
	// Disabled turns the server's detection off. It is sent as null.
	Disabled bool `json:"-"`
	// Type is TurnDetectionServerVAD or TurnDetectionSemanticVAD.
	Type string `json:"type,omitempty"`
	// Threshold, PrefixPaddingMs and SilenceDurationMs tune voice activity
	// detection.
	Threshold         *float64 `json:"threshold,omitempty"`
	PrefixPaddingMs   *int     `json:"prefix_padding_ms,omitempty"`
	SilenceDurationMs *int     `json:"silence_duration_ms,omitempty"`
	// Eagerness, CreateResponse and InterruptResponse tune semantic detection.
	Eagerness         string `json:"eagerness,omitempty"`
	CreateResponse    *bool  `json:"create_response,omitempty"`
	InterruptResponse *bool  `json:"interrupt_response,omitempty"`
}

// ServerVAD is voice activity detection at the server's documented defaults:
// a 0.5 threshold, 300 ms of padding before speech and 500 ms of silence to end
// it.
func ServerVAD() *TurnDetection {
	threshold, padding, silence := 0.5, 300, 500
	return &TurnDetection{
		Type: TurnDetectionServerVAD, Threshold: &threshold, PrefixPaddingMs: &padding, SilenceDurationMs: &silence,
	}
}

// SemanticVAD is detection by what the user said.
func SemanticVAD() *TurnDetection { return &TurnDetection{Type: TurnDetectionSemanticVAD} }

// NoTurnDetection turns the server's detection off.
func NoTurnDetection() *TurnDetection { return &TurnDetection{Disabled: true} }

// InputAudioNoiseReduction filters the user's audio for the kind of microphone:
// "near_field" for a close one, "far_field" for a distant one.
type InputAudioNoiseReduction struct {
	Type string `json:"type,omitempty"`
}

// AudioInput configures the user's audio.
type AudioInput struct {
	Format         *AudioFormat              `json:"format,omitempty"`
	Transcription  *InputAudioTranscription  `json:"transcription,omitempty"`
	NoiseReduction *InputAudioNoiseReduction `json:"noise_reduction,omitempty"`
	TurnDetection  *TurnDetection            `json:"turn_detection,omitempty"`
}

// MarshalJSON implements json.Marshaler, sending disabled turn detection as
// null.
func (a AudioInput) MarshalJSON() ([]byte, error) {
	type plain AudioInput
	if a.TurnDetection == nil || !a.TurnDetection.Disabled {
		return json.Marshal(plain(a))
	}
	a.TurnDetection = nil
	raw, err := json.Marshal(plain(a))
	if err != nil {
		return nil, err
	}
	var m map[string]json.RawMessage
	if err := json.Unmarshal(raw, &m); err != nil {
		return nil, err
	}
	m["turn_detection"] = json.RawMessage("null")
	return json.Marshal(m)
}

// AudioOutput configures the model's audio.
type AudioOutput struct {
	Format *AudioFormat `json:"format,omitempty"`
	// Voice is the voice the model speaks with.
	Voice string `json:"voice,omitempty"`
	// Speed is how fast it speaks.
	Speed *float64 `json:"speed,omitempty"`
}

// AudioConfiguration configures the audio both ways.
type AudioConfiguration struct {
	Input  *AudioInput  `json:"input,omitempty"`
	Output *AudioOutput `json:"output,omitempty"`
}

// Reasoning configures a reasoning-capable model.
type Reasoning struct {
	// Effort is how hard the model reasons: "minimal", "low", "medium", "high",
	// "xhigh", or any level the API adds later. Empty lets the server pick.
	Effort string `json:"effort,omitempty"`
}

// errTokenLimit reports a token limit that is neither a number nor "inf".
var errTokenLimit = errors.New(`events: token limit is neither a number nor "inf"`)

// TokenLimit bounds the tokens of a response: a number, or no bound at all.
type TokenLimit struct {
	n   int
	inf bool
}

// Tokens bounds a response to n tokens.
func Tokens(n int) *TokenLimit { return &TokenLimit{n: n} }

// Unlimited leaves a response unbounded.
func Unlimited() *TokenLimit { return &TokenLimit{inf: true} }

// MarshalJSON implements json.Marshaler.
func (l TokenLimit) MarshalJSON() ([]byte, error) {
	if l.inf {
		return []byte(`"inf"`), nil
	}
	return json.Marshal(l.n)
}

// UnmarshalJSON implements json.Unmarshaler.
func (l *TokenLimit) UnmarshalJSON(data []byte) error {
	var s string
	if json.Unmarshal(data, &s) == nil {
		if s != "inf" {
			return fmt.Errorf("%w: %q", errTokenLimit, s)
		}
		*l = TokenLimit{inf: true}
		return nil
	}
	var n int
	if err := json.Unmarshal(data, &n); err != nil {
		return err
	}
	*l = TokenLimit{n: n}
	return nil
}

// SessionProperties are the properties of a Realtime session. A field left at
// its zero value is not sent, leaving the server's own.
type SessionProperties struct {
	// Type is the session type; empty is sent as "realtime".
	Type string `json:"type,omitempty"`
	// Object and ID identify a session the server describes.
	Object string `json:"object,omitempty"`
	ID     string `json:"id,omitempty"`
	// Model is the realtime model. It is chosen when the session is opened and
	// cannot change while it runs.
	Model string `json:"model,omitempty"`
	// OutputModalities are what the model responds with.
	OutputModalities []Modality `json:"output_modalities,omitempty"`
	// Instructions are the system instructions.
	Instructions string `json:"instructions,omitempty"`
	// Audio configures the audio both ways.
	Audio *AudioConfiguration `json:"audio,omitempty"`
	// Tools are tools configured on the service, in the universal form. They
	// are not sent as they stand: a session update carries them, with the
	// built-in tools, in the API's form (see SessionUpdate).
	Tools frames.ToolsSchema `json:"-"`
	// WireTools are the tools in the API's form, as a session update carries
	// them and as the server describes them.
	WireTools []map[string]any `json:"tools,omitempty"`
	// ToolChoice is "auto", "none" or "required".
	ToolChoice string `json:"tool_choice,omitempty"`
	// MaxOutputTokens bounds each response.
	MaxOutputTokens *TokenLimit `json:"max_output_tokens,omitempty"`
	// Tracing is "auto" or a tracing configuration.
	Tracing any `json:"tracing,omitempty"`
	// Prompt references a prompt template and its variables.
	Prompt map[string]any `json:"prompt,omitempty"`
	// ExpiresAt is when the session expires.
	ExpiresAt *int64 `json:"expires_at,omitempty"`
	// Include names extra fields the server's events carry.
	Include []string `json:"include,omitempty"`
	// Reasoning configures a reasoning-capable model.
	Reasoning *Reasoning `json:"reasoning,omitempty"`
}

// MarshalJSON implements json.Marshaler, sending "realtime" for an empty type.
func (p SessionProperties) MarshalJSON() ([]byte, error) {
	type plain SessionProperties
	if p.Type == "" {
		p.Type = "realtime"
	}
	return json.Marshal(plain(p))
}

// Copy returns properties that can be changed without changing these: the
// nested configuration is copied, so a session update built from the copy
// leaves the stored properties as they were.
func (p SessionProperties) Copy() SessionProperties {
	out := p
	if p.Audio != nil {
		audio := *p.Audio
		if audio.Input != nil {
			in := *audio.Input
			audio.Input = &in
			if in.Transcription != nil {
				t := *in.Transcription
				t.Keywords = slices.Clone(t.Keywords)
				t.Languages = slices.Clone(t.Languages)
				audio.Input.Transcription = &t
			}
		}
		if audio.Output != nil {
			o := *audio.Output
			audio.Output = &o
		}
		out.Audio = &audio
	}
	if p.Reasoning != nil {
		r := *p.Reasoning
		out.Reasoning = &r
	}
	out.OutputModalities = append([]Modality(nil), p.OutputModalities...)
	return out
}

// InputTranscription is the transcription the properties configure, or nil.
func (p SessionProperties) InputTranscription() *InputAudioTranscription {
	if p.Audio == nil || p.Audio.Input == nil {
		return nil
	}
	return p.Audio.Input.Transcription
}

// TurnDetectionDisabled reports whether the properties turn the server's turn
// detection off. Unset is not off: the server then detects turns by default.
func (p SessionProperties) TurnDetectionDisabled() bool {
	return p.Audio != nil && p.Audio.Input != nil && p.Audio.Input.TurnDetection != nil &&
		p.Audio.Input.TurnDetection.Disabled
}
