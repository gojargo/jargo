// Package cartesia is a streaming text-to-speech service backed by Cartesia's
// TTS WebSocket. For each sentence it opens a connection, sends the transcript,
// and streams the raw PCM audio chunks downstream.
package cartesia

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"

	"github.com/coder/websocket"
	"github.com/gojargo/jargo/internal/validate"
	"github.com/gojargo/jargo/language"
	"github.com/gojargo/jargo/service/tts"
	"github.com/gojargo/jargo/service/wsutil"
	uctx "github.com/gojargo/jargo/utils/context"
)

// errProtocol is returned when Cartesia reports an error message.
//
//nolint:gochecknoglobals // sentinel error
var errProtocol = errors.New("cartesia: protocol error")

const (
	defaultURL = "wss://api.cartesia.ai/tts/websocket"
	// defaultVersion pins the Cartesia API version; it is sent on every request.
	defaultVersion = "2026-03-01"
	defaultModel   = "sonic-3.5"
	// defaultVoiceID is a public Cartesia voice.
	defaultVoiceID = "694f9389-aac1-45b6-b726-9d9369183238"
	// defaultSampleRate is the PCM rate jargo requests from Cartesia.
	defaultSampleRate = 24000
	defaultEncoding   = "pcm_s16le"
	defaultContainer  = "raw"
	// readLimit bounds a single WebSocket message; audio chunks arrive base64.
	readLimit = 1 << 20
)

// GenerationConfig guides Cartesia generation; it applies to sonic-3 and
// sonic-3.5 models. Fields left at their zero value are omitted.
type GenerationConfig struct {
	// Volume multiplies the generated speech volume (0.5 to 2.0).
	Volume *float64 `json:"volume,omitempty"`
	// Speed multiplies the speaking rate (0.6 to 1.5).
	Speed *float64 `json:"speed,omitempty"`
	// Emotion guides the emotional tone (e.g. "neutral", "excited", "sad").
	Emotion string `json:"emotion,omitempty"`
}

// Config configures the Cartesia TTS service.
type Config struct {
	// APIKey is the Cartesia API key. Required.
	APIKey string `validate:"required"`
	// URL overrides the TTS WebSocket endpoint; empty uses the hosted endpoint.
	URL string
	// Version sets the Cartesia-Version header; empty uses a pinned default.
	Version string
	// Model is the Cartesia model id; empty uses a default.
	Model string
	// VoiceID is the voice id; empty uses a default public voice.
	VoiceID string
	// Language for synthesis; the zero value leaves it unset (Cartesia defaults
	// to English). Mapped to Cartesia's base code.
	Language language.Language
	// SampleRate is the PCM rate requested from Cartesia and emitted downstream;
	// 0 uses 24 kHz.
	SampleRate int
	// Encoding is the audio encoding; empty uses "pcm_s16le".
	Encoding string
	// Container is the audio container; empty uses "raw".
	Container string
	// GenerationConfig guides generation (volume, speed, emotion) on supported
	// models; nil omits it.
	GenerationConfig *GenerationConfig
	// PronunciationDictID applies a custom pronunciation dictionary; empty omits it.
	PronunciationDictID string
	// WordTimestamps requests per-word timestamps and drives the word-aligned
	// text path: the TTS base emits a TTSTextFrame for each spoken word as its
	// audio plays, mapped back to its original written form, so the assistant
	// context records only what was actually spoken before an interruption. It is
	// off by default; leaving it off keeps the service behaving exactly as before.
	WordTimestamps bool
}

// Validate reports whether the configuration is usable.
func (c Config) Validate() error { return validate.Struct(c) }

// NewTTS builds a Cartesia TTS service.
func NewTTS(cfg Config) *tts.Base {
	if cfg.URL == "" {
		cfg.URL = defaultURL
	}
	if cfg.Version == "" {
		cfg.Version = defaultVersion
	}
	if cfg.Model == "" {
		cfg.Model = defaultModel
	}
	if cfg.VoiceID == "" {
		cfg.VoiceID = defaultVoiceID
	}
	if cfg.SampleRate == 0 {
		cfg.SampleRate = defaultSampleRate
	}
	if cfg.Encoding == "" {
		cfg.Encoding = defaultEncoding
	}
	if cfg.Container == "" {
		cfg.Container = defaultContainer
	}
	s := &synthesizer{cfg: cfg}
	if cfg.WordTimestamps {
		// Only the timestamp-aware type implements tts.WordTimestamps, so the base
		// takes the word-aligned path solely when the caller opts in.
		return tts.New("CartesiaTTS", &timedSynthesizer{synthesizer: s})
	}
	return tts.New("CartesiaTTS", s)
}

type synthesizer struct {
	cfg Config
}

// timedSynthesizer adds word-timestamp streaming on top of synthesizer. It
// implements tts.WordTimestamps.
type timedSynthesizer struct {
	*synthesizer
}

// SampleRate reports the requested PCM output rate.
func (s *synthesizer) SampleRate() int { return s.cfg.SampleRate }

// cartesiaLanguage maps a Language to Cartesia's language code: Cartesia wants
// the base code, returned only for languages it supports; otherwise "" (Cartesia
// defaults to English).
func cartesiaLanguage(l language.Language) string {
	switch base := l.BaseCode(); base {
	case "cs", "da", "de", "el", "en", "es", "fi", "fr", "gu", "he", "hi",
		"hr", "hu", "id", "it", "ja", "ka", "kn", "ko", "ml", "mr", "ms",
		"nl", "no", "pa", "pl", "pt", "ro", "ru", "sk", "sv", "ta", "te",
		"th", "tl", "tr", "uk", "vi", "zh":
		return base
	default:
		return ""
	}
}

// wsMessage is the subset of a Cartesia WebSocket message we read.
type wsMessage struct {
	Type           string         `json:"type"`
	Data           string         `json:"data"`
	Message        string         `json:"message"`
	WordTimestamps *wsWordTimings `json:"word_timestamps"`
}

// wsWordTimings is the payload of a Cartesia "timestamps" message: parallel
// arrays of spoken words and their start times, in seconds from the start of the
// synthesis.
type wsWordTimings struct {
	Words []string  `json:"words"`
	Start []float64 `json:"start"`
}

// Synthesize opens a session, sends the transcript, and streams audio chunks.
func (s *synthesizer) Synthesize(ctx context.Context, text string, emit func(pcm []byte) error) error {
	conn, err := s.dial(ctx)
	if err != nil {
		return err
	}
	defer func() { _ = conn.Close(websocket.StatusNormalClosure, "") }()

	if err := s.request(ctx, conn, text, false); err != nil {
		return err
	}
	return s.receive(ctx, conn, emit, nil)
}

// SynthesizeTimed streams audio and reports per-word timing, implementing
// tts.WordTimestamps. It requests timestamps and forwards each Cartesia
// "timestamps" message (after merging any punctuation-only tokens into the
// preceding word) to word.
func (s *timedSynthesizer) SynthesizeTimed(
	ctx context.Context,
	text string,
	emit func(pcm []byte) error,
	word func(text string, offset float64) error,
) error {
	conn, err := s.dial(ctx)
	if err != nil {
		return err
	}
	defer func() { _ = conn.Close(websocket.StatusNormalClosure, "") }()

	if err := s.request(ctx, conn, text, true); err != nil {
		return err
	}
	return s.receive(ctx, conn, emit, word)
}

func (s *synthesizer) dial(ctx context.Context) (*websocket.Conn, error) {
	header := http.Header{}
	header.Set("X-API-Key", s.cfg.APIKey)
	header.Set("Cartesia-Version", s.cfg.Version)
	return wsutil.Dial(ctx, s.cfg.URL, header, readLimit)
}

func (s *synthesizer) request(ctx context.Context, conn *websocket.Conn, text string, timestamps bool) error {
	msg := map[string]any{
		"model_id":   s.cfg.Model,
		"transcript": text,
		"voice":      map[string]any{"mode": "id", "id": s.cfg.VoiceID},
		"output_format": map[string]any{
			"container":   s.cfg.Container,
			"encoding":    s.cfg.Encoding,
			"sample_rate": s.cfg.SampleRate,
		},
		"context_id": "jargo",
		"continue":   false,
	}
	if timestamps {
		msg["add_timestamps"] = true
		msg["use_normalized_timestamps"] = false
	}
	if lang := cartesiaLanguage(s.cfg.Language); lang != "" {
		msg["language"] = lang
	}
	if s.cfg.GenerationConfig != nil {
		msg["generation_config"] = s.cfg.GenerationConfig
	}
	if s.cfg.PronunciationDictID != "" {
		msg["pronunciation_dict_id"] = s.cfg.PronunciationDictID
	}
	payload, err := json.Marshal(msg)
	if err != nil {
		return err
	}
	return conn.Write(ctx, websocket.MessageText, payload)
}

// receive reads audio chunks and, when word is non-nil, word-timestamp messages
// until the transcript is done.
func (s *synthesizer) receive(
	ctx context.Context,
	conn *websocket.Conn,
	emit func(pcm []byte) error,
	word func(text string, offset float64) error,
) error {
	for {
		_, data, err := conn.Read(ctx)
		if err != nil {
			return err
		}
		var m wsMessage
		if err := json.Unmarshal(data, &m); err != nil {
			continue
		}
		switch m.Type {
		case "chunk":
			pcm, err := base64.StdEncoding.DecodeString(m.Data)
			if err != nil {
				return err
			}
			if err := emit(pcm); err != nil {
				return err
			}
		case "timestamps":
			if word != nil {
				if err := emitWordTimings(m.WordTimestamps, word); err != nil {
					return err
				}
			}
		case "done":
			return nil
		case "error":
			return fmt.Errorf("%w: %s", errProtocol, m.Message)
		}
	}
}

// emitWordTimings merges punctuation-only tokens into the preceding word and
// forwards each resulting (word, start) pair to word.
func emitWordTimings(wt *wsWordTimings, word func(text string, offset float64) error) error {
	if wt == nil {
		return nil
	}
	batch := make([]uctx.WordTiming, 0, len(wt.Words))
	for i, w := range wt.Words {
		var start float64
		if i < len(wt.Start) {
			start = wt.Start[i]
		}
		batch = append(batch, uctx.WordTiming{Word: w, Offset: start})
	}
	for _, m := range uctx.MergePunctTokens(batch) {
		if err := word(m.Word, m.Offset); err != nil {
			return err
		}
	}
	return nil
}
