package soniox

import (
	"cmp"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"strings"
	"sync"

	"github.com/coder/websocket"
	"github.com/gojargo/jargo/language"
	"github.com/gojargo/jargo/service/settings"
	"github.com/gojargo/jargo/service/stt"
	"github.com/gojargo/jargo/service/wsutil"
)

// NewSTT builds a Soniox streaming STT service.
func NewSTT(cfg Config) *stt.StreamService {
	if cfg.URL == "" {
		cfg.URL = defaultURL
	}
	if cfg.Model == "" {
		cfg.Model = defaultModel
	}
	return stt.NewStream("SonioxSTT", &connector{cfg: cfg, live: newSTTSettings(cfg)}, cfg.SampleRate)
}

// ContextGeneralItem is one key-value pair of structured context.
type ContextGeneralItem struct {
	Key   string `json:"key"`
	Value string `json:"value"`
}

// ContextTranslationTerm maps an ambiguous or domain-specific term onto the
// translation it should take.
type ContextTranslationTerm struct {
	Source string `json:"source"`
	Target string `json:"target"`
}

// Context primes the model with what the conversation is about, for the models
// that take it.
type Context struct {
	// General is structured context as key-value pairs.
	General []ContextGeneralItem `json:"general,omitempty"`
	// Text is free-form context.
	Text string `json:"text,omitempty"`
	// Terms are words the model should expect to hear.
	Terms []string `json:"terms,omitempty"`
	// TranslationTerms fix how particular terms are translated.
	TranslationTerms []ContextTranslationTerm `json:"translation_terms,omitempty"`
}

// Settings is the part of the Soniox configuration that can change while the
// pipeline runs. Soniox is told all of it in the handshake that opens a session,
// so a change to any of it reopens the session.
type Settings struct {
	settings.STT

	// LanguageHints are the languages to expect, as Soniox names them.
	LanguageHints settings.Opt[[]string] `settings:"language_hints"`
	// LanguageHintsStrict confines recognition to the languages hinted.
	LanguageHintsStrict settings.Opt[bool] `settings:"language_hints_strict"`
	// Context primes the model with what the conversation is about.
	Context settings.Opt[Context] `settings:"context"`
	// EnableSpeakerDiarization labels speakers in the transcript.
	EnableSpeakerDiarization settings.Opt[bool] `settings:"enable_speaker_diarization"`
	// EnableLanguageIdentification reports the language of each token.
	EnableLanguageIdentification settings.Opt[bool] `settings:"enable_language_identification"`
	// MaxEndpointDelayMs bounds how long Soniox waits before ending a turn.
	MaxEndpointDelayMs settings.Opt[int] `settings:"max_endpoint_delay_ms"`
	// EndpointSensitivity tunes how readily Soniox calls a turn ended.
	EndpointSensitivity settings.Opt[float64] `settings:"endpoint_sensitivity"`
	// EndpointLatencyAdjustmentLevel trades endpoint latency against accuracy.
	EndpointLatencyAdjustmentLevel settings.Opt[int] `settings:"endpoint_latency_adjustment_level"`
	// ClientReferenceID labels the session in Soniox's own records.
	ClientReferenceID settings.Opt[string] `settings:"client_reference_id"`
}

// newSTTSettings is the starting state, taken from what the service was built
// with.
func newSTTSettings(cfg Config) *Settings {
	s := &Settings{}
	s.Model = settings.Set(cfg.Model)
	if hint := cfg.Language.BaseCode(); hint != "" {
		s.LanguageHints = settings.Set([]string{hint})
	}
	setOpt(&s.LanguageHintsStrict, cfg.LanguageHintsStrict)
	setOpt(&s.EnableSpeakerDiarization, cfg.EnableSpeakerDiarization)
	setOpt(&s.EnableLanguageIdentification, cfg.EnableLanguageIdentification)
	setOpt(&s.MaxEndpointDelayMs, cfg.MaxEndpointDelayMs)
	setOpt(&s.EndpointSensitivity, cfg.EndpointSensitivity)
	setOpt(&s.EndpointLatencyAdjustmentLevel, cfg.EndpointLatencyAdjustmentLevel)
	if cfg.ClientReferenceID != "" {
		s.ClientReferenceID = settings.Set(cfg.ClientReferenceID)
	}
	if cfg.Context != nil {
		s.Context = settings.Set(*cfg.Context)
	}
	return s
}

// setOpt gives o a value when the configuration carried one.
func setOpt[T any](o *settings.Opt[T], v *T) {
	if v != nil {
		*o = settings.Set(*v)
	}
}

type connector struct {
	cfg Config
	// live is what may change while the pipeline runs: the model and the
	// language, which is the set Soniox treats as changeable and that jargo's
	// configuration carries.
	live *Settings
}

// Metadata reports the transcript latency the turn strategies size their
// wait by.
func (c *connector) Metadata() stt.Metadata {
	return stt.Metadata{TTFSP99: cmp.Or(c.cfg.TTFSP99, stt.SonioxTTFSP99)}
}

// Settings is the configuration a caller may change while the pipeline runs.
func (c *connector) Settings() any { return c.live }

// UpdateSettings asks for the session to be reopened whenever anything changed.
// Soniox is told all of this in the handshake that opens the session, so a
// change reaches it only by opening another.
func (c *connector) UpdateSettings(context.Context, settings.Changed) (bool, error) {
	return true, nil
}

// ServiceLanguage names a language the way Soniox does, by its base code.
func (c *connector) ServiceLanguage(l language.Language) string {
	return l.BaseCode()
}

// Connect dials the WebSocket and sends the config handshake (which carries the
// API key).
func (c *connector) Connect(ctx context.Context, sampleRate int) (stt.Stream, error) {
	conn, err := wsutil.Dial(ctx, c.cfg.URL, nil, readLimit)
	if err != nil {
		return nil, err
	}

	if err := conn.Write(ctx, websocket.MessageText, c.config(sampleRate)); err != nil {
		_ = conn.Close(websocket.StatusInternalError, "config failed")
		return nil, err
	}
	return &stream{conn: conn, ctx: ctx}, nil
}

func (c *connector) config(sampleRate int) []byte {
	cfg := map[string]any{
		"api_key":                   c.cfg.APIKey,
		"model":                     c.live.Model.Or(c.cfg.Model),
		"audio_format":              "s16le",
		"sample_rate":               sampleRate,
		"num_channels":              1,
		"enable_endpoint_detection": c.cfg.EnableEndpointDetection == nil || *c.cfg.EnableEndpointDetection,
	}
	// Everything Soniox is told in the handshake and that may change while the
	// pipeline runs. A setting with no value is left out, so Soniox applies its
	// own default rather than being sent a zero that means something else.
	putOpt(cfg, "language_hints", c.live.LanguageHints)
	putOpt(cfg, "language_hints_strict", c.live.LanguageHintsStrict)
	putOpt(cfg, "context", c.live.Context)
	putOpt(cfg, "enable_speaker_diarization", c.live.EnableSpeakerDiarization)
	putOpt(cfg, "enable_language_identification", c.live.EnableLanguageIdentification)
	putOpt(cfg, "max_endpoint_delay_ms", c.live.MaxEndpointDelayMs)
	putOpt(cfg, "endpoint_sensitivity", c.live.EndpointSensitivity)
	putOpt(cfg, "endpoint_latency_adjustment_level", c.live.EndpointLatencyAdjustmentLevel)
	putOpt(cfg, "client_reference_id", c.live.ClientReferenceID)

	b, _ := json.Marshal(cfg) //nolint:errchkjson // map of known-serializable values
	return b
}

// putOpt writes a setting into the handshake only when it carries a value.
func putOpt[T any](cfg map[string]any, key string, o settings.Opt[T]) {
	if v, ok := o.Value(); ok {
		cfg[key] = v
	}
}

type stream struct {
	conn     *wsutil.Conn
	ctx      context.Context
	writeMu  sync.Mutex
	finalBuf string
}

// Send writes a chunk of PCM as a binary frame.
func (s *stream) Send(audio []byte) error {
	s.writeMu.Lock()
	defer s.writeMu.Unlock()
	return s.conn.Write(s.ctx, websocket.MessageBinary, audio)
}

// Recv reads the next batch of tokens. Final tokens accumulate; a "<end>" or
// "<fin>" marker flushes the utterance with EndOfTurn set. Provisional tokens
// surface as an interim result.
func (s *stream) Recv() ([]stt.Result, error) {
	for {
		_, data, err := s.conn.Read(s.ctx)
		if err != nil {
			return nil, err
		}
		var m message
		if json.Unmarshal(data, &m) != nil {
			continue
		}
		if m.ErrorCode != 0 {
			return nil, fmt.Errorf("%w: %s", errServer, m.ErrorMessage)
		}
		if m.Finished {
			return nil, io.EOF
		}

		var provisional strings.Builder
		endOfTurn := false
		for _, t := range m.Tokens {
			switch {
			case t.Text == endToken || t.Text == finToken:
				endOfTurn = endOfTurn || t.IsFinal
			case t.IsFinal:
				s.finalBuf += t.Text
			default:
				provisional.WriteString(t.Text)
			}
		}

		if endOfTurn && s.finalBuf != "" {
			text := strings.TrimSpace(s.finalBuf)
			s.finalBuf = ""
			return []stt.Result{{Text: text, Final: true, EndOfTurn: true}}, nil
		}
		if interim := s.finalBuf + provisional.String(); strings.TrimSpace(interim) != "" {
			return []stt.Result{{Text: strings.TrimSpace(interim), Final: false}}, nil
		}
	}
}

// Close signals end of audio and closes the socket.
func (s *stream) Close() error {
	s.writeMu.Lock()
	// An empty frame tells Soniox the audio is complete.
	_ = s.conn.Write(context.Background(), websocket.MessageBinary, []byte{})
	s.writeMu.Unlock()
	return s.conn.Close(websocket.StatusNormalClosure, "")
}

// message is the subset of a Soniox result message we use.
type message struct {
	Tokens       []token `json:"tokens"`
	Finished     bool    `json:"finished"`
	ErrorCode    int     `json:"error_code"`
	ErrorMessage string  `json:"error_message"`
}

type token struct {
	Text    string `json:"text"`
	IsFinal bool   `json:"is_final"`
}
