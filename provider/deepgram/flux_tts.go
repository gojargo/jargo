package deepgram

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/url"
	"slices"
	"strconv"

	"github.com/coder/websocket"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/internal/query"
	"github.com/gojargo/jargo/internal/validate"
	"github.com/gojargo/jargo/service/tts"
	"github.com/gojargo/jargo/service/wsutil"
)

const (
	// fluxSpeakURL is the streaming synthesis WebSocket.
	fluxSpeakURL = "wss://api.deepgram.com/v2/speak"
	// defaultFluxVoice is a current English Flux voice. Deepgram passes the
	// voice identifier as its "model" query parameter.
	defaultFluxVoice = "flux-alexis-en"
	// fluxTTSMsgSpeechMetadata is the definitive per-turn end signal, sent once
	// after all of a turn's audio.
	fluxTTSMsgSpeechMetadata = "SpeechMetadata"
	// fluxTTSReadLimit bounds a single incoming message. Audio frames can be
	// larger than the library's default, so raise the ceiling.
	fluxTTSReadLimit = 1 << 24
)

// errFluxUnsupportedRate is returned when a configured sample rate is not one
// Flux accepts.
//
//nolint:gochecknoglobals // sentinel error
var errFluxUnsupportedRate = errors.New("deepgram flux tts: unsupported sample rate")

// fluxTTSSampleRates lists the sample rates Flux synthesis accepts.
func fluxTTSSampleRates() []int { return []int{8000, 16000, 24000, 32000, 44100, 48000} }

// FluxTTSConfig configures the Flux streaming TTS service. The audio is always
// linear16 (raw 16-bit little-endian mono PCM), the format the pipeline uses.
type FluxTTSConfig struct {
	// APIKey is the Deepgram API key. Required.
	APIKey string `validate:"required"`
	// SpeakURL overrides the synthesis WebSocket endpoint; empty uses the hosted
	// endpoint.
	SpeakURL string
	// Voice is the Flux voice identifier (e.g. "flux-alexis-en"); empty uses a
	// default. It is sent as Deepgram's "model" query parameter.
	Voice string
	// SampleRate is the PCM rate requested and emitted downstream; 0 uses 24 kHz.
	// A non-zero rate must be one of 8000, 16000, 24000, 32000, 44100, 48000.
	SampleRate int
	// MipOptOut opts out of Deepgram's model-improvement program.
	MipOptOut *bool
	// Tag attaches billing tags to the request.
	Tag []string
	// ExtraQuery sets arbitrary additional query parameters; values override any
	// param of the same name set from other fields.
	ExtraQuery map[string]string
}

// Validate reports whether the configuration is usable.
func (c FluxTTSConfig) Validate() error {
	if err := validate.Struct(c); err != nil {
		return err
	}
	if c.SampleRate == 0 || slices.Contains(fluxTTSSampleRates(), c.SampleRate) {
		return nil
	}
	return fmt.Errorf("%w: %d", errFluxUnsupportedRate, c.SampleRate)
}

// withDefaults returns a copy of the config with empty fields filled in.
func (c FluxTTSConfig) withDefaults() FluxTTSConfig {
	if c.Voice == "" {
		c.Voice = defaultFluxVoice
	}
	if c.SampleRate == 0 {
		c.SampleRate = defaultTTSSampleRate
	}
	if c.SpeakURL == "" {
		c.SpeakURL = fluxSpeakURL
	}
	return c
}

// NewFluxTTS builds a Deepgram Flux streaming WebSocket TTS service. Each
// aggregated sentence is synthesized as one Flux turn: the text is sent and
// flushed, and the returned PCM is streamed downstream.
func NewFluxTTS(cfg FluxTTSConfig) *tts.Base {
	return tts.New("DeepgramFluxTTS", &fluxSynth{cfg: cfg.withDefaults()})
}

// fluxTTSQuery builds the synthesis query string.
func fluxTTSQuery(cfg FluxTTSConfig) url.Values {
	q := url.Values{}
	q.Set("model", cfg.Voice)
	q.Set("encoding", fluxEncoding)
	q.Set("sample_rate", strconv.Itoa(cfg.SampleRate))
	query.SetBoolOpt(q, "mip_opt_out", cfg.MipOptOut)
	query.AddAll(q, "tag", cfg.Tag)
	for k, v := range cfg.ExtraQuery {
		q.Set(k, v)
	}
	return q
}

// fluxSpeak is the message that sends text to synthesize.
type fluxSpeak struct {
	Type string `json:"type"`
	Text string `json:"text"`
}

// fluxTTSMessage is the subset of a Flux synthesis control message we consume.
type fluxTTSMessage struct {
	Type        string `json:"type"`
	Code        any    `json:"code"`
	Description string `json:"description"`
}

// errText renders a Warning/Error message for reporting.
func (m fluxTTSMessage) errText() string {
	switch {
	case m.Description != "" && m.Code != nil:
		return fmt.Sprintf("[%v] %s", m.Code, m.Description)
	case m.Description != "":
		return m.Description
	case m.Code != nil:
		return fmt.Sprintf("%v", m.Code)
	default:
		return "unknown error"
	}
}

type fluxSynth struct {
	cfg FluxTTSConfig
}

// SampleRate reports the requested PCM output rate.
func (s *fluxSynth) SampleRate() int { return s.cfg.SampleRate }

// Synthesize opens a synthesis session, sends text and a flush to render it as
// one turn, and streams the returned PCM to emit until the turn completes.
func (s *fluxSynth) RunTTS(ctx context.Context, text, _ string, yield func(f frames.Frame) error) error {
	emit := tts.PCMYielder(yield, s.SampleRate())
	q := fluxTTSQuery(s.cfg)

	header := http.Header{}
	header.Set("Authorization", "Token "+s.cfg.APIKey)

	conn, err := wsutil.Dial(ctx, s.cfg.SpeakURL+"?"+q.Encode(), header, fluxTTSReadLimit)
	if err != nil {
		return fmt.Errorf("deepgram flux tts: dial: %w", err)
	}
	defer func() { _ = conn.Close(websocket.StatusNormalClosure, "") }()

	speak, err := json.Marshal(fluxSpeak{Type: "Speak", Text: text})
	if err != nil {
		return fmt.Errorf("deepgram flux tts: encode: %w", err)
	}
	if err := conn.Write(ctx, websocket.MessageText, speak); err != nil {
		return fmt.Errorf("deepgram flux tts: speak: %w", err)
	}
	// Flush ends the turn so the server renders all remaining audio and then
	// sends SpeechMetadata.
	if err := conn.Write(ctx, websocket.MessageText, []byte(`{"type":"Flush"}`)); err != nil {
		return fmt.Errorf("deepgram flux tts: flush: %w", err)
	}

	for {
		typ, data, err := conn.Read(ctx)
		if err != nil {
			return fmt.Errorf("deepgram flux tts: recv: %w", err)
		}
		if typ == websocket.MessageBinary {
			if perr := emit(data); perr != nil {
				return perr
			}
			continue
		}
		done, cerr := fluxTTSControl(data)
		if cerr != nil {
			return cerr
		}
		if done {
			return nil
		}
	}
}

// fluxTTSControl inspects a JSON control message. It reports done when the turn
// is complete (SpeechMetadata) and an error on a fatal Error message.
func fluxTTSControl(data []byte) (bool, error) {
	var m fluxTTSMessage
	if json.Unmarshal(data, &m) == nil {
		switch m.Type {
		case fluxTTSMsgSpeechMetadata:
			return true, nil
		case fluxMsgError:
			return false, fmt.Errorf("%w: %s", errFluxServer, m.errText())
		}
	}
	// Non-JSON frames and informational control messages (Connected,
	// SpeechStarted, Flushed, SessionMetadata, Warning): no action.
	return false, nil
}
