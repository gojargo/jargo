package assemblyai

import (
	"context"
	"encoding/json"
	"net/http"
	"net/url"
	"strconv"
	"sync"

	"github.com/coder/websocket"
	"github.com/gojargo/jargo/internal/query"
	"github.com/gojargo/jargo/language"
	"github.com/gojargo/jargo/service/stt"
	"github.com/gojargo/jargo/service/wsutil"
)

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

	query.SetStrOpt(q, "speech_model", cfg.Model)
	if code := assemblyaiLanguage(cfg.Language); code != "" {
		q.Set("language_code", code)
	}
	query.SetBoolOpt(q, "language_detection", cfg.LanguageDetection)
	query.SetBoolTrue(q, "format_turns", cfg.FormatTurns)
	query.SetBoolOpt(q, "formatted_finals", cfg.FormattedFinals)
	query.SetIntOpt(q, "word_finalization_max_wait_time", cfg.WordFinalizationMaxWaitTime)
	query.SetFloatOpt(q, "end_of_turn_confidence_threshold", cfg.EndOfTurnConfidenceThreshold)
	query.SetIntOpt(q, "min_turn_silence", cfg.MinTurnSilence)
	query.SetIntOpt(q, "max_turn_silence", cfg.MaxTurnSilence)
	query.SetStrOpt(q, "prompt", cfg.Prompt)
	query.SetBoolOpt(q, "speaker_labels", cfg.SpeakerLabels)
	query.SetFloatOpt(q, "vad_threshold", cfg.VADThreshold)
	query.SetStrOpt(q, "domain", cfg.Domain)

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

	conn, err := wsutil.Dial(ctx, c.cfg.BaseURL+"?"+q.Encode(), header, readLimit)
	if err != nil {
		return nil, err
	}
	return &stream{conn: conn, ctx: ctx}, nil
}

type stream struct {
	conn    *wsutil.Conn
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
