package cartesia

import (
	"cmp"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"net/url"
	"strconv"
	"strings"
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
	defaultSTTVersion  = "2026-03-01"
	defaultSTTModel    = "ink-whisper"
	defaultSTTEncoding = "pcm_s16le"
	// defaultSTTLanguage is Cartesia's default transcription language.
	defaultSTTLanguage = "en"

	// Cartesia caps a connection at 100 keyterms totaling 1200 characters.
	maxKeyterms     = 100
	maxKeytermChars = 1200
	// Only the ink-2 model family honors keyterms.
	keytermModelPrefix = "ink-2"

	// Cartesia closes a connection that has been idle for three minutes. The
	// timeout resets on every message sent, so silence is submitted well inside
	// the window rather than at the edge of it.
	sttKeepaliveTimeout  = 120 * time.Second
	sttKeepaliveInterval = 30 * time.Second
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
	// Keyterm biases transcription towards the given terms or phrases. Cartesia
	// binds them to a connection and only the ink-2 model family honors them, so
	// changing them while the pipeline runs reopens the session.
	Keyterm []string

	// TTFSP99 overrides the measured transcript latency the turn strategies
	// size their wait by; 0 uses stt.CartesiaTTFSP99.
	TTFSP99 time.Duration
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
	return stt.NewStream("CartesiaSTT", newSTTConnector(cfg), cfg.SampleRate)
}

type sttConnector struct {
	cfg STTConfig
	// live is what may change while the pipeline runs: the model and the
	// language, which is the set Cartesia treats as changeable.
	live *STTSettings
}

// newSTTConnector builds the connector with its settings seeded, which is the
// only way it should be built: the settings store is not optional.
func newSTTConnector(cfg STTConfig) *sttConnector {
	return &sttConnector{cfg: cfg, live: newSTTSettings(cfg)}
}

// STTSettings is the part of the Cartesia STT configuration that can change
// while the pipeline runs. Cartesia takes all of it as query parameters when the
// session opens, so a change to any of it reopens the session.
type STTSettings struct {
	settings.STT

	// Keyterm biases transcription towards the given terms or phrases, sent as
	// repeated keyterm query parameters. Cartesia binds keyterms to a
	// connection, and only the ink-2 model family honors them.
	Keyterm settings.Opt[[]string] `settings:"keyterm"`
}

// newSTTSettings is the starting state, taken from what the service was built
// with.
func newSTTSettings(cfg STTConfig) *STTSettings {
	s := &STTSettings{}
	s.Model = settings.Set(cfg.Model)
	s.Language = settings.Set(cartesiaSTTLanguage(cfg.Language))
	if len(cfg.Keyterm) > 0 {
		s.Keyterm = settings.Set(cfg.Keyterm)
	}
	return s
}

// Keepalive holds an idle session open. Cartesia closes a connection that has
// carried nothing for three minutes, and the timeout resets on every message
// sent, so silence well inside that window is enough.
func (c *sttConnector) Keepalive() stt.KeepaliveOptions {
	return stt.KeepaliveOptions{Timeout: sttKeepaliveTimeout, Interval: sttKeepaliveInterval}
}

// Metadata reports Cartesia's time-to-final-segment latency to downstream
// processors.
func (c *sttConnector) Metadata() stt.Metadata {
	return stt.Metadata{TTFSP99: cmp.Or(c.cfg.TTFSP99, stt.CartesiaTTFSP99), Model: c.live.Model.Or(c.cfg.Model)}
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
	model := c.live.Model.Or(c.cfg.Model)
	q := url.Values{}
	q.Set("model", model)
	q.Set("language", c.live.Language.Or(cartesiaSTTLanguage(c.cfg.Language)))
	q.Set("encoding", c.cfg.Encoding)
	q.Set("sample_rate", strconv.Itoa(sampleRate))
	for _, term := range keytermsFor(model, c.live.Keyterm.Or(nil)) {
		q.Add("keyterm", term)
	}
	return c.cfg.URL + "?" + encodeQuery(q)
}

// keytermsFor is the keyterms to send with a session on the given model: none
// unless the model honors them, and no more than Cartesia accepts.
func keytermsFor(model string, keyterms []string) []string {
	prepared := prepareKeyterms(keyterms)
	if len(prepared) == 0 {
		return nil
	}
	if !strings.HasPrefix(model, keytermModelPrefix) {
		slog.Warn("keyterms are ignored on this model",
			"service", "CartesiaSTT", "model", model, "supported", keytermModelPrefix)
		return nil
	}
	return prepared
}

// prepareKeyterms drops blank terms and truncates the list to what Cartesia
// accepts on one connection, warning about what it drops. Truncating keeps an
// oversized list from failing the connection partway through a call.
func prepareKeyterms(keyterms []string) []string {
	var prepared []string
	var chars int
	var seen int
	for _, term := range keyterms {
		term = strings.TrimSpace(term)
		if term == "" {
			continue
		}
		seen++
		if len(prepared) >= maxKeyterms || chars+len(term) > maxKeytermChars {
			continue
		}
		prepared = append(prepared, term)
		chars += len(term)
	}
	if dropped := seen - len(prepared); dropped > 0 {
		slog.Warn("dropping keyterms beyond what Cartesia accepts",
			"max_keyterms", maxKeyterms, "max_chars", maxKeytermChars, "dropped", dropped)
	}
	return prepared
}

// encodeQuery encodes q with spaces as %20. Cartesia expects that inside a
// keyterm, and Go's own encoding writes a space as a plus.
func encodeQuery(q url.Values) string {
	return strings.ReplaceAll(q.Encode(), "+", "%20")
}

type sttStream struct {
	conn    *wsutil.Conn
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

// Finalize tells Cartesia the speech has ended, so it flushes the transcript for
// what it has rather than waiting on its own endpointing. The session carries on
// afterwards for the next utterance.
func (s *sttStream) Finalize() error {
	s.writeMu.Lock()
	defer s.writeMu.Unlock()
	return s.conn.Write(s.ctx, websocket.MessageText, []byte("finalize"))
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
		case fieldTranscript:
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
