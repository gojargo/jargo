package kyutai

import (
	"cmp"
	"context"
	"net/http"
	"strings"
	"sync"

	"github.com/coder/websocket"
	"github.com/gojargo/jargo/audio/resample"
	"github.com/gojargo/jargo/service/stt"
	"github.com/gojargo/jargo/service/wsutil"
	"github.com/vmihailenco/msgpack/v5"
)

const (
	// defaultSTTURL is the moshi-server streaming ASR endpoint on localhost.
	defaultSTTURL = "ws://127.0.0.1:8080/api/asr-streaming"
	// sttFrameSamples is the number of 24 kHz samples per Audio message the server
	// expects (80 ms).
	sttFrameSamples = 1920
	// pausePredictionHead selects which look-ahead window of the semantic-VAD
	// pause prediction to threshold; index 2 is the 2 s window the reference
	// client uses.
	pausePredictionHead = 2
	// pauseThreshold is the pause-prediction confidence above which the user is
	// considered to have finished their turn.
	pauseThreshold = 0.5
)

// NewSTT builds a Kyutai streaming STT service backed by moshi-server.
func NewSTT(cfg Config) *stt.StreamService {
	if cfg.URL == "" {
		cfg.URL = defaultSTTURL
	}
	if cfg.APIKey == "" {
		cfg.APIKey = defaultToken
	}
	return stt.NewStream("KyutaiSTT", &connector{cfg: cfg}, cfg.SampleRate)
}

type connector struct {
	cfg Config
}

// Metadata reports the transcript latency the turn strategies size their
// wait by.
// The server runs wherever it is hosted, so it carries no measurement of its own:
// measure yours and set TTFSP99.
func (c *connector) Metadata() stt.Metadata {
	return stt.Metadata{TTFSP99: cmp.Or(c.cfg.TTFSP99, stt.DefaultTTFSP99)}
}

// Connect dials the moshi-server ASR WebSocket and prepares a resampler from the
// pipeline's input rate to the server's 24 kHz.
func (c *connector) Connect(ctx context.Context, sampleRate int) (stt.Stream, error) {
	header := http.Header{}
	header.Set("kyutai-api-key", c.cfg.APIKey)

	conn, err := wsutil.Dial(ctx, c.cfg.URL, header, wsutil.DefaultReadLimit)
	if err != nil {
		return nil, err
	}
	rs, err := resample.New(sampleRate, defaultSampleRate, 1)
	if err != nil {
		_ = conn.Close(websocket.StatusInternalError, "resampler")
		return nil, err
	}
	return &stream{conn: conn, ctx: ctx, rs: rs}, nil
}

type stream struct {
	conn *wsutil.Conn
	ctx  context.Context
	rs   *resample.Resampler

	// Send side (pipeline goroutine): writeMu guards writes against Close, and
	// pending holds resampled samples awaiting a full 1920-sample frame.
	writeMu sync.Mutex
	pending []float32

	// Receive side (read-loop goroutine): the running utterance and whether a
	// word has arrived since the last pause.
	utterance     strings.Builder
	speechStarted bool
}

// Send resamples the S16LE PCM to 24 kHz, converts it to float32, and forwards
// it to the server in 1920-sample Audio messages.
func (s *stream) Send(audio []byte) error {
	s.pending = int16BytesToFloat32(s.pending, s.rs.Process(audio))
	n := 0
	for len(s.pending)-n >= sttFrameSamples {
		if err := s.sendAudio(s.pending[n : n+sttFrameSamples]); err != nil {
			return err
		}
		n += sttFrameSamples
	}
	s.pending = append(s.pending[:0], s.pending[n:]...)
	return nil
}

func (s *stream) sendAudio(pcm []float32) error {
	b, err := msgpack.Marshal(map[string]any{msgTypeKey: "Audio", "pcm": pcm})
	if err != nil {
		return err
	}
	s.writeMu.Lock()
	defer s.writeMu.Unlock()
	return s.conn.Write(s.ctx, websocket.MessageBinary, b)
}

// Recv reads the server's result messages, returning cumulative interim transcripts as
// words arrive and one finalized end-of-turn transcript when the semantic VAD
// predicts a pause.
func (s *stream) Recv() ([]stt.Result, error) {
	for {
		_, data, err := s.conn.Read(s.ctx)
		if err != nil {
			return nil, err
		}
		var m sttMsg
		if err := msgpack.Unmarshal(data, &m); err != nil {
			continue
		}
		if r, ok := s.handle(m); ok {
			return []stt.Result{r}, nil
		}
	}
}

// handle folds one STT message into the running utterance and reports the
// result to emit, if any. A "Word" extends the utterance and yields a cumulative
// interim; a "Step" whose pause prediction crosses the threshold ends the turn
// and yields the finalized utterance.
func (s *stream) handle(m sttMsg) (stt.Result, bool) {
	switch m.Type {
	case "Word":
		if m.Text == "" {
			return stt.Result{}, false
		}
		if s.utterance.Len() > 0 {
			s.utterance.WriteByte(' ')
		}
		s.utterance.WriteString(m.Text)
		s.speechStarted = true
		return stt.Result{Text: s.utterance.String()}, true
	case "Step":
		if !s.speechStarted || len(m.Prs) <= pausePredictionHead ||
			m.Prs[pausePredictionHead] <= pauseThreshold {
			return stt.Result{}, false
		}
		text := s.utterance.String()
		s.utterance.Reset()
		s.speechStarted = false
		if text == "" {
			return stt.Result{}, false
		}
		return stt.Result{Text: text, Final: true, EndOfTurn: true}, true
	default:
		return stt.Result{}, false
	}
}

// Close releases the resampler and closes the socket.
func (s *stream) Close() error {
	if s.rs != nil {
		s.rs.Close()
	}
	return s.conn.Close(websocket.StatusNormalClosure, "")
}
