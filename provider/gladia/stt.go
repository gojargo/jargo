package gladia

import (
	"bytes"
	"cmp"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"maps"
	"net/http"
	"net/url"
	"sync"

	"github.com/coder/websocket"
	"github.com/gojargo/jargo/processor/turns"
	"github.com/gojargo/jargo/service/stt"
	"github.com/gojargo/jargo/service/wsutil"
	errs "github.com/gojargo/jargo/utils/errors"
)

// NewSTT builds a Gladia streaming STT service. It works best behind a turn
// detector: Gladia finalizes per utterance rather than per turn.
func NewSTT(cfg Config) *stt.StreamService {
	cfg = withDefaults(cfg)
	c := &connector{cfg: cfg, http: &http.Client{}}
	svc := stt.NewStream("GladiaSTT", c, cfg.SampleRate)
	// The connector is handed the service it drives, so it can report a failure
	// Gladia attaches to a message without ending the session. There is no call
	// left to fail by the time it arrives on the read loop, so the service is
	// the only way back.
	c.svc = svc
	return svc
}

// withDefaults fills in what the service supplies for a caller who left it
// unset. The audio shape has to be sent on every session, so it defaults to what
// the pipeline produces rather than being left empty.
func withDefaults(cfg Config) Config {
	if cfg.URL == "" {
		cfg.URL = liveURL
	}
	if cfg.Encoding == "" {
		cfg.Encoding = defaultEncoding
	}
	if cfg.BitDepth == 0 {
		cfg.BitDepth = defaultBitDepth
	}
	if cfg.Channels == 0 {
		cfg.Channels = defaultChannels
	}
	if cfg.Model == "" {
		cfg.Model = defaultModel
	}
	return cfg
}

type connector struct {
	cfg  Config
	http *http.Client
	// svc is the service this connector drives, for reporting a problem the
	// session survives.
	svc *stt.StreamService
}

// reportProblem puts a problem the session survived on the pipeline as a
// non-fatal error, so an application hears about it while transcription
// carries on over the same connection.
func (c *connector) reportProblem(ctx context.Context, msg string) {
	if c.svc == nil {
		return
	}
	c.svc.PushError(ctx, msg, nil, false)
}

// Metadata describes the service downstream. With Gladia's own detection driving
// the turn, the service reports when the user starts and stops speaking, so the
// aggregator is asked to defer to those reports rather than running its own
// detection alongside them. Without it the pipeline's defaults stand.
func (c *connector) Metadata() stt.Metadata {
	m := stt.Metadata{TTFSP99: cmp.Or(c.cfg.TTFSP99, stt.GladiaTTFSP99), Model: c.cfg.Model}
	if c.cfg.EnableVAD {
		m.UserTurnStrategies = turns.ExternalStrategies(turns.ExternalStrategiesConfig{
			EnableInterruptions: c.cfg.InterruptOnSpeech,
		})
	}
	return m
}

// Connect initializes a session over REST then dials the returned WebSocket.
func (c *connector) Connect(ctx context.Context, sampleRate int) (stt.Stream, error) {
	wsURL, err := c.initSession(ctx, sampleRate)
	if err != nil {
		return nil, err
	}
	conn, err := wsutil.Dial(ctx, wsURL, nil, readLimit)
	if err != nil {
		return nil, err
	}
	return &stream{
		conn:   conn,
		ctx:    ctx,
		vad:    c.cfg.EnableVAD,
		report: c.reportProblem,
	}, nil
}

// settings builds the session-init body for the given sample rate.
func (cfg *Config) settings(sampleRate int) map[string]any {
	s := map[string]any{
		"encoding":    cfg.Encoding,
		"sample_rate": sampleRate,
		"bit_depth":   cfg.BitDepth,
		"channels":    cfg.Channels,
		"model":       cfg.Model,
	}
	if cfg.Endpointing != nil {
		s["endpointing"] = *cfg.Endpointing
	}
	if cfg.MaximumDurationWithoutEndpointing != nil {
		s["maximum_duration_without_endpointing"] = *cfg.MaximumDurationWithoutEndpointing
	}
	if cfg.LanguageConfig != nil {
		s["language_config"] = cfg.LanguageConfig
	}
	if cfg.PreProcessing != nil {
		s["pre_processing"] = cfg.PreProcessing
	}
	if len(cfg.RealtimeProcessing) > 0 {
		s["realtime_processing"] = cfg.RealtimeProcessing
	}
	if len(cfg.CustomMetadata) > 0 {
		s["custom_metadata"] = cfg.CustomMetadata
	}
	mc := cfg.MessagesConfig
	if mc == nil {
		on := true
		mc = &MessagesConfig{ReceivePartialTranscripts: &on, ReceiveFinalTranscripts: &on}
	}
	if cfg.EnableVAD && mc.ReceiveSpeechEvents == nil {
		// The turn is driven by Gladia's own detection, so the messages that
		// report it have to be asked for. A caller who set the field themselves
		// is left alone, including one who deliberately turned it off.
		on := true
		filtered := *mc
		filtered.ReceiveSpeechEvents = &on
		mc = &filtered
	}
	s["messages_config"] = mc
	maps.Copy(s, cfg.ExtraSettings)
	return s
}

func (c *connector) initSession(ctx context.Context, sampleRate int) (string, error) {
	body, err := json.Marshal(c.cfg.settings(sampleRate))
	if err != nil {
		return "", err
	}
	endpoint := c.cfg.URL
	if c.cfg.Region != "" {
		endpoint += "?" + url.Values{"region": {c.cfg.Region}}.Encode()
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(body))
	if err != nil {
		return "", err
	}
	req.Header.Set("x-gladia-key", c.cfg.APIKey)
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.http.Do(req)
	if err != nil {
		return "", err
	}
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode/100 != 2 {
		msg, _ := io.ReadAll(io.LimitReader(resp.Body, 512))
		return "", errs.NewHTTPStatusError(resp.StatusCode, fmt.Errorf("%w %d: %s", errStatus, resp.StatusCode, msg))
	}
	var out struct {
		URL string `json:"url"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return "", err
	}
	return out.URL, nil
}

type stream struct {
	conn    *wsutil.Conn
	ctx     context.Context
	writeMu sync.Mutex
	// vad is whether Gladia's own detection drives the turn, which is what makes
	// its speech boundaries something to act on.
	vad bool
	// report puts a problem the session survives on the pipeline; nil drops it.
	report func(ctx context.Context, msg string)
}

// message is the subset of a Gladia session message we read.
type message struct {
	Type string `json:"type"`
	// Acknowledged is set on an audio chunk Gladia accepted.
	Acknowledged bool `json:"acknowledged"`
	// Error is the failure Gladia attaches to a message it could not act on.
	Error *messageError `json:"error"`
	// Data is the message's payload; nil when Gladia sent none, as it does
	// alongside an error.
	Data *messageData `json:"data"`
}

// messageData is the payload of a transcript message.
type messageData struct {
	IsFinal   bool `json:"is_final"`
	Utterance struct {
		Text     string `json:"text"`
		Language string `json:"language"`
	} `json:"utterance"`
}

// messageError is the failure Gladia attaches to a message.
type messageError struct {
	Message string `json:"message"`
}

// errorMessage is the failure's message, or a stand-in when Gladia gave none.
func (m message) errorMessage() string {
	if m.Error == nil || m.Error.Message == "" {
		return "Unknown error"
	}
	return m.Error.Message
}

// Send writes a chunk of PCM audio as a binary frame.
func (s *stream) Send(audio []byte) error {
	s.writeMu.Lock()
	defer s.writeMu.Unlock()
	return s.conn.Write(s.ctx, websocket.MessageBinary, audio)
}

// Recv reads the next transcript message and maps it to a result.
func (s *stream) Recv() ([]stt.Result, error) {
	for {
		_, data, err := s.conn.Read(s.ctx)
		if err != nil {
			return nil, err
		}
		var m message
		if err := json.Unmarshal(data, &m); err != nil {
			continue
		}
		if r, ok := s.result(m); ok {
			return []stt.Result{r}, nil
		}
	}
}

// result maps one message to a result, reporting whether it carried anything.
// The speech boundaries are only acted on when Gladia's own detection is what
// drives the turn; otherwise the pipeline's detection decides and a boundary
// from here would compete with it.
func (s *stream) result(m message) (stt.Result, bool) {
	switch m.Type {
	case msgAudioChunk:
		// An acknowledged chunk needs nothing: the audio is not held for
		// resending. One that was not is a chunk Gladia rejected.
		if !m.Acknowledged {
			s.reportProblem("Gladia rejected an audio chunk: " + m.errorMessage())
		}
		return stt.Result{}, false
	case msgTranslation:
		// A failed translation carries an error and no payload. It is the
		// translation that failed, not the session, so transcription carries on.
		if m.Error != nil || m.Data == nil {
			s.reportProblem("Gladia translation addon error: " + m.errorMessage())
		}
		return stt.Result{}, false
	case msgTranscript:
		if m.Data == nil || m.Data.Utterance.Text == "" {
			return stt.Result{}, false
		}
		return stt.Result{
			Text:      m.Data.Utterance.Text,
			Final:     m.Data.IsFinal,
			EndOfTurn: m.Data.IsFinal,
			Language:  m.Data.Utterance.Language,
		}, true
	case msgSpeechStart:
		if !s.vad {
			return stt.Result{}, false
		}
		return stt.Result{Speech: stt.SpeechStarted}, true
	case msgSpeechEnd:
		if !s.vad {
			return stt.Result{}, false
		}
		return stt.Result{Speech: stt.SpeechStopped}, true
	}
	return stt.Result{}, false
}

// reportProblem reports a problem the session survives.
func (s *stream) reportProblem(msg string) {
	if s.report != nil {
		s.report(s.ctx, msg)
	}
}

// Close stops the session and closes the socket.
func (s *stream) Close() error {
	s.writeMu.Lock()
	_ = s.conn.Write(context.Background(), websocket.MessageText, []byte(`{"type":"stop_recording"}`))
	s.writeMu.Unlock()
	return s.conn.Close(websocket.StatusNormalClosure, "")
}
