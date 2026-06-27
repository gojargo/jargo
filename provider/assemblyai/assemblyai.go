// Package assemblyai is a streaming speech-to-text service backed by
// AssemblyAI's Universal-Streaming (v3) WebSocket. It streams audio and pushes
// InterimTranscriptionFrames and finalized TranscriptionFrames downstream.
//
// A turn is finalized once AssemblyAI reports a formatted end-of-turn, which
// marks the end of the user's speech.
package assemblyai

import (
	"context"
	"encoding/json"
	"net/http"
	"net/url"
	"strconv"
	"sync"

	"github.com/coder/websocket"
	"github.com/gojargo/jargo/internal/validate"
	"github.com/gojargo/jargo/language"
	"github.com/gojargo/jargo/service/stt"
)

const (
	wsBase = "wss://streaming.assemblyai.com/v3/ws"
	// defaultEncoding is the audio encoding AssemblyAI expects from jargo.
	defaultEncoding = "pcm_s16le"
	// readLimit bounds a single WebSocket message; long turns carry many words.
	readLimit = 1 << 20
)

// Config configures the AssemblyAI STT service. Optional fields modeled as
// pointers or slices are omitted from the request when unset.
type Config struct {
	// APIKey is the AssemblyAI API key. Required.
	APIKey string `validate:"required"`
	// BaseURL overrides the streaming WebSocket endpoint; empty uses the hosted
	// endpoint.
	BaseURL string
	// SampleRate is the input audio sample rate; 0 uses the transport's rate.
	SampleRate int
	// Encoding is the audio encoding; empty uses "pcm_s16le".
	Encoding string
	// Model selects the speech model (sent as speech_model); empty uses the
	// account default.
	Model string
	// Language declares the audio language (sent as language_code); the zero value
	// leaves it unset. Mapped to AssemblyAI's base code.
	Language language.Language
	// LanguageDetection enables automatic language detection; nil omits it.
	// Mutually exclusive with Language.
	LanguageDetection *bool
	// FormatTurns formats finalized turns (punctuation/casing); nil defaults to true.
	FormatTurns *bool
	// FormattedFinals formats partial finals; nil omits it.
	FormattedFinals *bool
	// WordFinalizationMaxWaitTime caps the wait in ms for word finalization;
	// nil omits it.
	WordFinalizationMaxWaitTime *int
	// EndOfTurnConfidenceThreshold sets the confidence needed to end a turn
	// (0.0-1.0); nil omits it.
	EndOfTurnConfidenceThreshold *float64
	// MinTurnSilence is the minimum silence in ms to end a turn when confident;
	// nil omits it.
	MinTurnSilence *int
	// MaxTurnSilence is the maximum silence in ms before forcing a turn end;
	// nil omits it.
	MaxTurnSilence *int
	// KeytermsPrompt boosts recognition of the given terms; empty omits it.
	KeytermsPrompt []string
	// Prompt steers transcription with free-text guidance; empty omits it.
	Prompt string
	// SpeakerLabels enables speaker diarization; nil omits it.
	SpeakerLabels *bool
	// VADThreshold sets the VAD confidence threshold (0.0-1.0); nil omits it.
	VADThreshold *float64
	// Domain hints the transcription domain; empty omits it.
	Domain string
	// ExtraQuery sets arbitrary additional query parameters not modeled above
	// (e.g. U3 Pro-only options); values override any param of the same name.
	ExtraQuery map[string]string
}

// Validate reports whether the configuration is usable.
func (cfg Config) Validate() error { return validate.Struct(cfg) }

// NewSTT builds an AssemblyAI streaming STT service.
func NewSTT(cfg Config) *stt.StreamService {
	if cfg.BaseURL == "" {
		cfg.BaseURL = wsBase
	}
	if cfg.Encoding == "" {
		cfg.Encoding = defaultEncoding
	}
	return stt.NewStream("AssemblyAISTT", &connector{cfg: cfg}, cfg.SampleRate)
}

type connector struct {
	cfg Config
}

// query builds the streaming query string for the given sample rate.
func (cfg *Config) query(sampleRate int) url.Values {
	q := url.Values{}
	q.Set("sample_rate", strconv.Itoa(sampleRate))
	q.Set("encoding", cfg.Encoding)

	setStrOpt(q, "speech_model", cfg.Model)
	if code := assemblyaiLanguage(cfg.Language); code != "" {
		q.Set("language_code", code)
	}
	setBoolOpt(q, "language_detection", cfg.LanguageDetection)
	setBoolTrue(q, "format_turns", cfg.FormatTurns)
	setBoolOpt(q, "formatted_finals", cfg.FormattedFinals)
	setIntOpt(q, "word_finalization_max_wait_time", cfg.WordFinalizationMaxWaitTime)
	setFloatOpt(q, "end_of_turn_confidence_threshold", cfg.EndOfTurnConfidenceThreshold)
	setIntOpt(q, "min_turn_silence", cfg.MinTurnSilence)
	setIntOpt(q, "max_turn_silence", cfg.MaxTurnSilence)
	setStrOpt(q, "prompt", cfg.Prompt)
	setBoolOpt(q, "speaker_labels", cfg.SpeakerLabels)
	setFloatOpt(q, "vad_threshold", cfg.VADThreshold)
	setStrOpt(q, "domain", cfg.Domain)

	if len(cfg.KeytermsPrompt) > 0 {
		if b, err := json.Marshal(cfg.KeytermsPrompt); err == nil {
			q.Set("keyterms_prompt", string(b))
		}
	}
	for k, v := range cfg.ExtraQuery {
		q.Set(k, v)
	}
	return q
}

// setBoolTrue sets key to v, defaulting to true when v is nil.
func setBoolTrue(q url.Values, key string, v *bool) {
	val := true
	if v != nil {
		val = *v
	}
	q.Set(key, strconv.FormatBool(val))
}

// setBoolOpt sets key only when v is non-nil.
func setBoolOpt(q url.Values, key string, v *bool) {
	if v != nil {
		q.Set(key, strconv.FormatBool(*v))
	}
}

// setIntOpt sets key only when v is non-nil.
func setIntOpt(q url.Values, key string, v *int) {
	if v != nil {
		q.Set(key, strconv.Itoa(*v))
	}
}

// setFloatOpt sets key only when v is non-nil.
func setFloatOpt(q url.Values, key string, v *float64) {
	if v != nil {
		q.Set(key, strconv.FormatFloat(*v, 'g', -1, 64))
	}
}

// setStrOpt sets key only when v is non-empty.
func setStrOpt(q url.Values, key, v string) {
	if v != "" {
		q.Set(key, v)
	}
}

// assemblyaiLanguage maps a Language to AssemblyAI's language_code: it wants the
// base code, returned only for languages it supports; otherwise "" (unset).
func assemblyaiLanguage(l language.Language) string {
	switch base := l.BaseCode(); base {
	case "en", "es", "fr", "de", "it", "pt", "tr", "nl", "sv", "no", "da",
		"fi", "hi", "vi", "ar", "he", "ja", "ur", "zh":
		return base
	default:
		return ""
	}
}

// Connect opens the streaming WebSocket for the given sample rate.
func (c *connector) Connect(ctx context.Context, sampleRate int) (stt.Stream, error) {
	q := c.cfg.query(sampleRate)

	header := http.Header{}
	header.Set("Authorization", c.cfg.APIKey)

	conn, resp, err := websocket.Dial(ctx, c.cfg.BaseURL+"?"+q.Encode(), &websocket.DialOptions{HTTPHeader: header})
	if resp != nil && resp.Body != nil {
		_ = resp.Body.Close()
	}
	if err != nil {
		return nil, err
	}
	conn.SetReadLimit(readLimit)
	return &stream{conn: conn, ctx: ctx}, nil
}

type stream struct {
	conn    *websocket.Conn
	ctx     context.Context
	writeMu sync.Mutex
}

// turnMessage is the subset of a v3 Turn message we read.
type turnMessage struct {
	Type            string `json:"type"`
	Transcript      string `json:"transcript"`
	EndOfTurn       bool   `json:"end_of_turn"`
	TurnIsFormatted bool   `json:"turn_is_formatted"`
}

// Send writes a chunk of PCM audio as a binary frame.
func (s *stream) Send(audio []byte) error {
	s.writeMu.Lock()
	defer s.writeMu.Unlock()
	return s.conn.Write(s.ctx, websocket.MessageBinary, audio)
}

// Recv reads the next Turn message and maps it to a result. A turn is final only
// once it is both ended and formatted; the unformatted end-of-turn and ongoing
// turns are interim.
func (s *stream) Recv() ([]stt.Result, error) {
	for {
		_, data, err := s.conn.Read(s.ctx)
		if err != nil {
			return nil, err
		}
		var m turnMessage
		if err := json.Unmarshal(data, &m); err != nil {
			continue
		}
		if m.Type != "Turn" || m.Transcript == "" {
			continue
		}
		final := m.EndOfTurn && m.TurnIsFormatted
		return []stt.Result{{Text: m.Transcript, Final: final, EndOfTurn: final}}, nil
	}
}

// Close terminates the session and closes the socket.
func (s *stream) Close() error {
	s.writeMu.Lock()
	_ = s.conn.Write(context.Background(), websocket.MessageText, []byte(`{"type":"Terminate"}`))
	s.writeMu.Unlock()
	return s.conn.Close(websocket.StatusNormalClosure, "")
}
