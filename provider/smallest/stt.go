package smallest

import (
	"cmp"
	"context"
	"encoding/json"
	"log/slog"
	"net/http"
	"net/url"
	"strconv"
	"sync"
	"time"

	"github.com/coder/websocket"
	"github.com/gojargo/jargo/internal/validate"
	"github.com/gojargo/jargo/service/stt"
	"github.com/gojargo/jargo/service/wsutil"
)

const (
	defaultSTTURL      = "wss://api.smallest.ai"
	defaultSTTEncoding = "linear16"
	defaultSTTModel    = "pulse"
	defaultSTTNumerals = "auto"
)

// STTConfig configures the Smallest AI streaming STT (Pulse) service.
type STTConfig struct {
	// APIKey is the Smallest AI API key. Required.
	APIKey string `validate:"required"`
	// BaseURL overrides the STT WebSocket base; empty uses the hosted endpoint.
	BaseURL string
	// Model is the transcription model; empty uses "pulse".
	Model string
	// Language is the audio language code; empty lets the service auto-detect.
	Language string
	// Encoding is the PCM encoding sent to the API; empty uses "linear16".
	Encoding string
	// SampleRate is the input audio sample rate; 0 uses the transport's rate.
	SampleRate int
	// WordTimestamps asks for word-level timings on each transcript.
	WordTimestamps bool
	// FullTranscript asks each message to carry the transcript so far rather
	// than only what is new.
	FullTranscript bool
	// SentenceTimestamps asks for sentence-level timings on each transcript.
	SentenceTimestamps bool
	// RedactPII redacts personally identifiable information.
	RedactPII bool
	// RedactPCI redacts payment card information.
	RedactPCI bool
	// Numerals writes spoken numbers as digits; empty uses "auto".
	Numerals string
	// Diarize labels each transcript with the speaker it came from.
	Diarize bool
	// Endpointing finalizes a transcript promptly on trailing silence; nil
	// defaults to true.
	Endpointing *bool
	// Keywords biases recognition towards domain terms, as comma-separated
	// KEYWORD:INTENSIFIER pairs (e.g. "NVIDIA:2"). Empty sends none, since an
	// empty value would register one empty keyword.
	Keywords string
	// Format punctuates and capitalizes the transcripts; nil defaults to true.
	Format *bool

	// TTFSP99 overrides the measured transcript latency the turn strategies
	// size their wait by; 0 uses stt.SmallestTTFSP99.
	TTFSP99 time.Duration
}

// Validate reports whether the configuration is usable.
func (c STTConfig) Validate() error { return validate.Struct(c) }

// NewSTT builds a Smallest AI streaming STT service.
func NewSTT(cfg STTConfig) *stt.StreamService {
	if cfg.BaseURL == "" {
		cfg.BaseURL = defaultSTTURL
	}
	if cfg.Encoding == "" {
		cfg.Encoding = defaultSTTEncoding
	}
	if cfg.Model == "" {
		cfg.Model = defaultSTTModel
	}
	if cfg.Numerals == "" {
		cfg.Numerals = defaultSTTNumerals
	}
	return stt.NewStream("SmallestSTT", &connector{cfg: cfg}, cfg.SampleRate)
}

type connector struct {
	cfg STTConfig
}

// Metadata reports the model in use and the transcript latency the turn
// strategies size their wait by.
func (c *connector) Metadata() stt.Metadata {
	return stt.Metadata{TTFSP99: cmp.Or(c.cfg.TTFSP99, stt.SmallestTTFSP99), Model: c.cfg.Model}
}

// query builds the session's query string for the given input sample rate.
func (c *connector) query(sampleRate int) url.Values {
	q := url.Values{}
	q.Set("model", c.cfg.Model)
	if c.cfg.Language != "" {
		q.Set("language", c.cfg.Language)
	}
	q.Set("encoding", c.cfg.Encoding)
	q.Set("sample_rate", strconv.Itoa(sampleRate))
	q.Set("word_timestamps", strconv.FormatBool(c.cfg.WordTimestamps))
	q.Set("full_transcript", strconv.FormatBool(c.cfg.FullTranscript))
	q.Set("sentence_timestamps", strconv.FormatBool(c.cfg.SentenceTimestamps))
	q.Set("redact_pii", strconv.FormatBool(c.cfg.RedactPII))
	q.Set("redact_pci", strconv.FormatBool(c.cfg.RedactPCI))
	q.Set("numerals", c.cfg.Numerals)
	q.Set("diarize", strconv.FormatBool(c.cfg.Diarize))
	q.Set("endpointing", strconv.FormatBool(boolOr(c.cfg.Endpointing, true)))
	q.Set("format", strconv.FormatBool(boolOr(c.cfg.Format, true)))
	// An empty value would register one empty keyword, so send the parameter
	// only when there is something to bias towards.
	if c.cfg.Keywords != "" {
		q.Set("keywords", c.cfg.Keywords)
	}
	return q
}

// boolOr returns *p, or def when p is nil.
func boolOr(p *bool, def bool) bool {
	if p == nil {
		return def
	}
	return *p
}

// Connect opens a Pulse streaming session for the given input sample rate.
func (c *connector) Connect(ctx context.Context, sampleRate int) (stt.Stream, error) {
	endpoint := c.cfg.BaseURL + "/waves/v1/stt/live?" + c.query(sampleRate).Encode()

	header := http.Header{}
	header.Set("Authorization", "Bearer "+c.cfg.APIKey)
	header.Set("X-Source", "jargo")

	conn, err := wsutil.Dial(ctx, endpoint, header, wsutil.DefaultReadLimit)
	if err != nil {
		return nil, err
	}
	return &sttStream{ctx: ctx, conn: conn}, nil
}

type sttStream struct {
	conn    *wsutil.Conn
	ctx     context.Context
	writeMu sync.Mutex
}

// sttMessage is the subset of a Pulse transcription message we read.
type sttMessage struct {
	IsFinal    bool   `json:"is_final"`
	IsLast     bool   `json:"is_last"`
	Transcript string `json:"transcript"`
	Language   string `json:"language"`
}

// Send writes a chunk of 16-bit mono PCM to the session.
func (s *sttStream) Send(audio []byte) error {
	s.writeMu.Lock()
	defer s.writeMu.Unlock()
	return s.conn.Write(s.ctx, websocket.MessageBinary, audio)
}

// Finalize flushes the transcript for the utterance that just ended. The session
// stays open for the next one, so there is no reconnection to pay for between
// turns.
func (s *sttStream) Finalize() error {
	s.writeMu.Lock()
	defer s.writeMu.Unlock()
	return s.conn.Write(s.ctx, websocket.MessageText, []byte(`{"type":"finalize"}`))
}

// Recv reads the next transcription; a final transcript marks the end of turn.
func (s *sttStream) Recv() ([]stt.Result, error) {
	for {
		_, data, err := s.conn.Read(s.ctx)
		if err != nil {
			return nil, err
		}
		var m sttMessage
		if err := json.Unmarshal(data, &m); err != nil {
			continue
		}
		if m.IsLast {
			// Sent once the stream is closed at the end of the call: the
			// service is about to close the session itself.
			slog.Debug("smallest stt received the last message of the session")
		}
		if m.Transcript == "" {
			continue
		}
		return []stt.Result{{
			Text:      m.Transcript,
			Final:     m.IsFinal,
			EndOfTurn: m.IsFinal,
			Language:  m.Language,
		}}, nil
	}
}

// Close tears down the session.
func (s *sttStream) Close() error {
	return s.conn.Close(websocket.StatusNormalClosure, "")
}
