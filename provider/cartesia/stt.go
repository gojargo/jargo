package cartesia

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/url"
	"strconv"
	"sync"
	"time"

	"github.com/coder/websocket"
	"github.com/gojargo/jargo/internal/validate"
	"github.com/gojargo/jargo/language"
	"github.com/gojargo/jargo/service/settings"
	"github.com/gojargo/jargo/service/stt"
	"github.com/gojargo/jargo/service/wsutil"
)

// errSTTProtocol is returned when Cartesia reports an STT error message.
//
//nolint:gochecknoglobals // sentinel error
var errSTTProtocol = errors.New("cartesia: stt protocol error")

const (
	defaultSTTURL = "wss://api.cartesia.ai/stt/websocket"
	// defaultSTTVersion pins the Cartesia STT API version; sent on every request.
	defaultSTTVersion  = "2025-04-16"
	defaultSTTModel    = "ink-whisper"
	defaultSTTEncoding = "pcm_s16le"
	// defaultSTTLanguage is Cartesia's default transcription language.
	defaultSTTLanguage = "en"
	// sttTTFSP99 is the time-to-final-segment P99 latency reported to downstream.
	sttTTFSP99 = 810 * time.Millisecond
)

// STTConfig configures the Cartesia streaming STT service.
type STTConfig struct {
	// APIKey is the Cartesia API key, sent as the X-API-Key header. Required.
	APIKey string `validate:"required"`
	// URL overrides the STT WebSocket endpoint; empty uses the hosted endpoint.
	URL string
	// Version sets the Cartesia-Version header; empty uses a pinned default.
	Version string
	// Model is the transcription model; empty uses a default.
	Model string
	// Encoding is the audio encoding; empty uses "pcm_s16le".
	Encoding string
	// Language hints the spoken language; the zero value defaults to English.
	// Mapped to Cartesia's base code.
	Language language.Language
	// SampleRate is the input audio sample rate; 0 uses the transport's rate.
	SampleRate int
}

// Validate reports whether the configuration is usable.
func (c STTConfig) Validate() error { return validate.Struct(c) }

// NewSTT builds a Cartesia streaming STT service.
func NewSTT(cfg STTConfig) *stt.StreamService {
	if cfg.URL == "" {
		cfg.URL = defaultSTTURL
	}
	if cfg.Version == "" {
		cfg.Version = defaultSTTVersion
	}
	if cfg.Model == "" {
		cfg.Model = defaultSTTModel
	}
	if cfg.Encoding == "" {
		cfg.Encoding = defaultSTTEncoding
	}
	return stt.NewStream("CartesiaSTT", &sttConnector{cfg: cfg, live: newSTTSettings(cfg)}, cfg.SampleRate)
}

type sttConnector struct {
	cfg STTConfig
	// live is what may change while the pipeline runs: the model and the
	// language, which is the set Cartesia treats as changeable.
	live *settings.STT
}

// newSTTSettings is the starting state, taken from what the service was built
// with.
func newSTTSettings(cfg STTConfig) *settings.STT {
	s := &settings.STT{}
	s.Model = settings.Set(cfg.Model)
	s.Language = settings.Set(cartesiaSTTLanguage(cfg.Language))
	return s
}

// Metadata reports Cartesia's time-to-final-segment latency to downstream
// processors.
func (c *sttConnector) Metadata() stt.Metadata {
	return stt.Metadata{TTFSP99: sttTTFSP99, Model: c.live.Model.Or(c.cfg.Model)}
}

// Settings is the configuration a caller may change while the pipeline runs.
func (c *sttConnector) Settings() any { return c.live }

// UpdateSettings asks for the session to be reopened whenever anything changed.
// Cartesia takes the model and the language as query parameters when the session
// opens, so a change reaches it only by opening another.
func (c *sttConnector) UpdateSettings(context.Context, settings.Changed) (bool, error) {
	return true, nil
}

// ServiceLanguage names a language the way Cartesia does, by its base code.
func (c *sttConnector) ServiceLanguage(l language.Language) string {
	return cartesiaSTTLanguage(l)
}

// cartesiaSTTLanguage maps a Language to Cartesia's STT language code. Cartesia
// wants the base code; the zero value defaults to English.
func cartesiaSTTLanguage(l language.Language) string {
	if base := l.BaseCode(); base != "" {
		return base
	}
	return defaultSTTLanguage
}

// Connect dials the STT WebSocket, passing the model and audio parameters as
// query arguments and the API key in the header.
func (c *sttConnector) Connect(ctx context.Context, sampleRate int) (stt.Stream, error) {
	header := http.Header{}
	header.Set("X-API-Key", c.cfg.APIKey)
	header.Set("Cartesia-Version", c.cfg.Version)

	conn, err := wsutil.Dial(ctx, c.endpoint(sampleRate), header, wsutil.DefaultReadLimit)
	if err != nil {
		return nil, err
	}
	return &sttStream{conn: conn, ctx: ctx}, nil
}

// endpoint builds the STT WebSocket URL with the model, language, encoding, and
// sample rate query parameters.
func (c *sttConnector) endpoint(sampleRate int) string {
	q := url.Values{}
	q.Set("model", c.live.Model.Or(c.cfg.Model))
	q.Set("language", c.live.Language.Or(cartesiaSTTLanguage(c.cfg.Language)))
	q.Set("encoding", c.cfg.Encoding)
	q.Set("sample_rate", strconv.Itoa(sampleRate))
	return c.cfg.URL + "?" + q.Encode()
}

type sttStream struct {
	conn    *websocket.Conn
	ctx     context.Context
	writeMu sync.Mutex
}

// sttMessage is the subset of a Cartesia STT message we read.
type sttMessage struct {
	Type     string `json:"type"`
	Text     string `json:"text"`
	IsFinal  bool   `json:"is_final"`
	Language string `json:"language"`
	Message  string `json:"message"`
}

// Send writes a chunk of PCM as a binary frame.
func (s *sttStream) Send(audio []byte) error {
	s.writeMu.Lock()
	defer s.writeMu.Unlock()
	return s.conn.Write(s.ctx, websocket.MessageBinary, audio)
}

// Recv reads the next transcript message, surfacing interim and final results.
// The server derives turn boundaries from the caller's audio stream, so finals
// are left for the upstream turn detector to close.
func (s *sttStream) Recv() ([]stt.Result, error) {
	for {
		_, data, err := s.conn.Read(s.ctx)
		if err != nil {
			return nil, err
		}
		var m sttMessage
		if json.Unmarshal(data, &m) != nil {
			continue
		}
		switch m.Type {
		case "transcript":
			if m.Text == "" {
				continue
			}
			return []stt.Result{{Text: m.Text, Final: m.IsFinal, Language: m.Language}}, nil
		case "error":
			return nil, fmt.Errorf("%w: %s", errSTTProtocol, m.Message)
		}
	}
}

// Close signals end of audio and closes the socket.
func (s *sttStream) Close() error {
	s.writeMu.Lock()
	// "done" tells Cartesia the audio stream is complete.
	_ = s.conn.Write(context.Background(), websocket.MessageText, []byte("done"))
	s.writeMu.Unlock()
	return s.conn.Close(websocket.StatusNormalClosure, "")
}
