package cartesia

import (
	"cmp"
	"context"
	"encoding/json"
	"errors"
	"log/slog"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"sync"
	"time"
	"unicode/utf8"

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

	// Cartesia closes a connection that has been idle for three minutes. The
	// timeout resets on every message sent, so silence is submitted well inside
	// the window rather than at the edge of it.
	sttKeepaliveTimeout  = 120 * time.Second
	sttKeepaliveInterval = 30 * time.Second
)

// keytermModelPrefixes are the model families that honor keyterms.
//
//nolint:gochecknoglobals // a lookup table
var keytermModelPrefixes = []string{"ink-2", "ink-preview"}

// honorsKeyterms reports whether the model belongs to a family that honors
// keyterms.
func honorsKeyterms(model string) bool {
	for _, prefix := range keytermModelPrefixes {
		if strings.HasPrefix(model, prefix) {
			return true
		}
	}
	return false
}

// STTConfig configures the Cartesia streaming STT service.
type STTConfig struct {
	// APIKey is the Cartesia API key, sent as the X-API-Key header. Required.
	APIKey string `validate:"required"`
	// URL overrides the STT WebSocket endpoint; empty uses the hosted endpoint.
	URL string
	// Version sets the Cartesia-Version header; empty uses a pinned default.
	Version string
	// Headers are sent on the WebSocket handshake, on top of the ones the
	// service sets, for a deployment that authorizes or routes its own way.
	Headers map[string]string
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
	// binds them to a connection and only the ink-2 and ink-preview model
	// families honor them, so changing them while the pipeline runs reopens the
	// session.
	Keyterm []string
	// MinVolume is the volume, between 0 and 1, below which audio counts as
	// silence for Cartesia's own finalization of a transcript; nil leaves it
	// to Cartesia. Only ink-whisper models honor it.
	MinVolume *float64
	// MaxSilenceDurationSecs is how many seconds of silence Cartesia waits
	// before finalizing a transcript on its own; nil leaves it to Cartesia.
	// Only ink-whisper models honor it.
	MaxSilenceDurationSecs *float64

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
	c := newSTTConnector(cfg)
	svc := stt.NewStream("CartesiaSTT", c, cfg.SampleRate)
	c.svc = svc
	return svc
}

type sttConnector struct {
	cfg STTConfig
	// live is what may change while the pipeline runs: the model and the
	// language, which is the set Cartesia treats as changeable.
	live *STTSettings
	// svc is the service this connector drives, for reporting the errors the
	// server sends on a session that carries on.
	svc *stt.StreamService
}

// reportServerError puts an error the server sent on the pipeline as a
// non-fatal error frame, carrying the server's message. The session carries
// on over the same connection.
func (c *sttConnector) reportServerError(ctx context.Context, msg string) {
	if c.svc == nil {
		slog.Warn("cartesia stt reported an error", "message", msg)
		return
	}
	c.svc.PushError(ctx, msg, nil, false)
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
	// connection, and only the ink-2 and ink-preview model families honor
	// them.
	Keyterm settings.Opt[[]string] `settings:"keyterm"`
	// MinVolume is the volume below which audio counts as silence for
	// Cartesia's own finalization. Only ink-whisper models honor it.
	MinVolume settings.Opt[float64] `settings:"min_volume"`
	// MaxSilenceDurationSecs is how many seconds of silence Cartesia waits
	// before finalizing on its own. Only ink-whisper models honor it.
	MaxSilenceDurationSecs settings.Opt[float64] `settings:"max_silence_duration_secs"`
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
	if cfg.MinVolume != nil {
		s.MinVolume = settings.Set(*cfg.MinVolume)
	}
	if cfg.MaxSilenceDurationSecs != nil {
		s.MaxSilenceDurationSecs = settings.Set(*cfg.MaxSilenceDurationSecs)
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
	// Applied last, so a deployment can route or authorize the handshake its own
	// way, overriding what the service sets.
	for k, v := range c.cfg.Headers {
		header.Set(k, v)
	}

	conn, err := wsutil.Dial(ctx, c.endpoint(sampleRate), header, wsutil.DefaultReadLimit)
	if err != nil {
		return nil, err
	}
	return &sttStream{conn: conn, ctx: ctx, report: c.reportServerError}, nil
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
	c.addFinalization(q, model)
	return c.cfg.URL + "?" + encodeQuery(q)
}

// finalizationModelPrefix names the model family that honors the finalization
// settings.
const finalizationModelPrefix = "ink-whisper"

// addFinalization adds the settings tuning Cartesia's own finalization of a
// transcript, when any is set, for a model that honors them. Another model
// would not, so they are left out with a warning.
func (c *sttConnector) addFinalization(q url.Values, model string) {
	var names []string
	set := map[string]float64{}
	if v, ok := c.live.MinVolume.Value(); ok {
		names, set["min_volume"] = append(names, "min_volume"), v
	}
	if v, ok := c.live.MaxSilenceDurationSecs.Value(); ok {
		names, set["max_silence_duration_secs"] = append(names, "max_silence_duration_secs"), v
	}
	if len(names) == 0 {
		return
	}
	if !strings.HasPrefix(model, finalizationModelPrefix) {
		slog.Warn("these settings are only supported on "+finalizationModelPrefix+" models; ignoring them",
			"settings", strings.Join(names, ", "), "model", model)
		return
	}
	for _, name := range names {
		q.Set(name, pythonFloat(set[name]))
	}
}

// pythonFloat formats a float the way Cartesia's documented examples are
// written, a whole number keeping its ".0".
func pythonFloat(v float64) string {
	s := strconv.FormatFloat(v, 'f', -1, 64)
	if !strings.ContainsAny(s, ".eE") {
		s += ".0"
	}
	return s
}

// keytermsFor is the keyterms to send with a session on the given model: none
// unless the model honors them, and no more than Cartesia accepts.
func keytermsFor(model string, keyterms []string) []string {
	prepared := prepareKeyterms(keyterms)
	if len(prepared) == 0 {
		return nil
	}
	if !honorsKeyterms(model) {
		slog.Warn("keyterms are ignored on this model",
			"service", "CartesiaSTT", "model", model,
			"supported", strings.Join(keytermModelPrefixes, ", "))
		return nil
	}
	return prepared
}

// prepareKeyterms drops blank terms and truncates the list to what Cartesia
// accepts on one connection, warning about what it drops. Truncating keeps an
// oversized list from failing the connection partway through a call.
func prepareKeyterms(keyterms []string) []string {
	var terms []string
	for _, term := range keyterms {
		if term = strings.TrimSpace(term); term != "" {
			terms = append(terms, term)
		}
	}
	// The list is cut at the first term that does not fit, so what is sent is
	// always a prefix of what was asked for. Characters are counted, not bytes.
	var prepared []string
	var chars int
	for _, term := range terms {
		n := utf8.RuneCountInString(term)
		if len(prepared) >= maxKeyterms || chars+n > maxKeytermChars {
			break
		}
		prepared = append(prepared, term)
		chars += n
	}
	seen := len(terms)
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
	// report puts an error the server sent on the pipeline.
	report func(ctx context.Context, msg string)
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
			// An error the server reports in band leaves the session open: it is
			// reported, and reading carries on.
			s.report(s.ctx, m.Message)
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
