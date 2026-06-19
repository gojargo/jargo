// Package deepgram provides Deepgram's streaming speech-to-text service (over
// the live transcription WebSocket) and its Aura text-to-speech service.
//
// The STT service pushes InterimTranscriptionFrames and finalized
// TranscriptionFrames downstream; a finalized transcript with Deepgram's
// speech_final marks the end of the user's turn.
package deepgram

import (
	"context"
	"encoding/json"
	"net/http"
	"net/url"
	"os"
	"strconv"
	"sync"
	"time"

	"github.com/coder/websocket"
	"github.com/gojargo/jargo/service/stt"
)

const (
	listenURL       = "wss://api.deepgram.com/v1/listen"
	keepAlivePeriod = 8 * time.Second
	defaultSTTModel = "nova-3"
	defaultLanguage = "en-US"
)

// Config configures the STT service.
type Config struct {
	// APIKey is the Deepgram API key; empty uses the DEEPGRAM_API_KEY env var.
	APIKey string
	// Model is the Deepgram model; empty uses "nova-3".
	Model string
	// Language is the transcription language; empty uses "en-US".
	Language string
	// SampleRate is the input audio sample rate; 0 uses the transport's rate.
	SampleRate int
}

// NewSTT builds a Deepgram streaming STT service.
func NewSTT(cfg Config) *stt.StreamService {
	if cfg.APIKey == "" {
		cfg.APIKey = os.Getenv("DEEPGRAM_API_KEY")
	}
	if cfg.Model == "" {
		cfg.Model = defaultSTTModel
	}
	if cfg.Language == "" {
		cfg.Language = defaultLanguage
	}
	return stt.NewStream("DeepgramSTT", &connector{cfg: cfg}, cfg.SampleRate)
}

type connector struct {
	cfg Config
}

// Connect dials the live transcription WebSocket for the given sample rate.
func (c *connector) Connect(ctx context.Context, sampleRate int) (stt.Stream, error) {
	q := url.Values{}
	q.Set("model", c.cfg.Model)
	q.Set("language", c.cfg.Language)
	q.Set("encoding", "linear16")
	q.Set("sample_rate", strconv.Itoa(sampleRate))
	q.Set("channels", "1")
	q.Set("interim_results", "true")
	q.Set("smart_format", "true")
	q.Set("punctuate", "true")
	q.Set("endpointing", "300")
	q.Set("utterance_end_ms", "1000")
	q.Set("vad_events", "true")

	header := http.Header{}
	header.Set("Authorization", "Token "+c.cfg.APIKey)

	conn, resp, err := websocket.Dial(ctx, listenURL+"?"+q.Encode(), &websocket.DialOptions{HTTPHeader: header})
	if resp != nil && resp.Body != nil {
		_ = resp.Body.Close()
	}
	if err != nil {
		return nil, err
	}
	s := &stream{conn: conn, ctx: ctx}
	s.wg.Go(s.keepAlive)
	return s, nil
}

type stream struct {
	conn    *websocket.Conn
	ctx     context.Context
	writeMu sync.Mutex
	wg      sync.WaitGroup
}

// dgMessage is the subset of Deepgram's live transcription result we use.
type dgMessage struct {
	Type    string `json:"type"`
	Channel struct {
		Alternatives []struct {
			Transcript string `json:"transcript"`
		} `json:"alternatives"`
	} `json:"channel"`
	IsFinal     bool `json:"is_final"`
	SpeechFinal bool `json:"speech_final"`
}

// Send writes a chunk of PCM audio as a binary frame.
func (s *stream) Send(audio []byte) error {
	s.writeMu.Lock()
	defer s.writeMu.Unlock()
	return s.conn.Write(s.ctx, websocket.MessageBinary, audio)
}

// Recv reads the next result. A finalized result carries Deepgram's speech_final
// as the end-of-turn signal.
func (s *stream) Recv() ([]stt.Result, error) {
	for {
		_, data, err := s.conn.Read(s.ctx)
		if err != nil {
			return nil, err
		}
		var m dgMessage
		if err := json.Unmarshal(data, &m); err != nil {
			continue
		}
		if m.Type != "Results" || len(m.Channel.Alternatives) == 0 {
			continue
		}
		text := m.Channel.Alternatives[0].Transcript
		if text == "" {
			continue
		}
		return []stt.Result{{Text: text, Final: m.IsFinal, EndOfTurn: m.SpeechFinal}}, nil
	}
}

// keepAlive sends a periodic KeepAlive so Deepgram does not close an idle
// connection during silence.
func (s *stream) keepAlive() {
	ticker := time.NewTicker(keepAlivePeriod)
	defer ticker.Stop()
	for {
		select {
		case <-s.ctx.Done():
			return
		case <-ticker.C:
			s.writeMu.Lock()
			_ = s.conn.Write(s.ctx, websocket.MessageText, []byte(`{"type":"KeepAlive"}`))
			s.writeMu.Unlock()
		}
	}
}

// Close asks Deepgram to flush and then closes the socket.
func (s *stream) Close() error {
	s.writeMu.Lock()
	_ = s.conn.Write(context.Background(), websocket.MessageText, []byte(`{"type":"CloseStream"}`))
	s.writeMu.Unlock()
	s.wg.Wait()
	return s.conn.Close(websocket.StatusNormalClosure, "")
}
