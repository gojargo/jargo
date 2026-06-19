// Package deepgram is a streaming speech-to-text service backed by Deepgram's
// live transcription WebSocket. It sends received audio to Deepgram and pushes
// InterimTranscriptionFrames and (finalized) TranscriptionFrames downstream.
//
// A finalized TranscriptionFrame (Deepgram's speech_final) marks the end of the
// user's turn, which the user aggregator uses to trigger the LLM.
package deepgram

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"
	"os"
	"strconv"
	"sync"
	"time"

	"github.com/coder/websocket"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor"
)

const (
	listenURL       = "wss://api.deepgram.com/v1/listen"
	keepAlivePeriod = 8 * time.Second
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

// Service is a Deepgram streaming STT processor.
type Service struct {
	*processor.Base
	cfg        Config
	sampleRate int

	conn   *websocket.Conn
	connMu sync.Mutex // serializes writes to conn

	wg     sync.WaitGroup
	cancel context.CancelFunc
}

// New builds a Deepgram STT service.
func New(cfg Config) *Service {
	if cfg.APIKey == "" {
		cfg.APIKey = os.Getenv("DEEPGRAM_API_KEY")
	}
	if cfg.Model == "" {
		cfg.Model = "nova-3"
	}
	if cfg.Language == "" {
		cfg.Language = "en-US"
	}
	s := &Service{cfg: cfg}
	s.Base = processor.New("DeepgramSTT", s)
	return s
}

// ProcessFrame manages the connection lifecycle and streams audio.
func (s *Service) ProcessFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	if err := s.Base.ProcessFrame(ctx, f, dir); err != nil {
		return err
	}
	switch fr := f.(type) {
	case *frames.StartFrame:
		if err := s.PushFrame(ctx, f, dir); err != nil {
			return err
		}
		s.sampleRate = s.cfg.SampleRate
		if s.sampleRate == 0 {
			s.sampleRate = fr.AudioInSampleRate
		}
		return s.connect(ctx)
	case *frames.InputAudioRawFrame:
		s.sendAudio(ctx, fr.Audio)
		return s.PushFrame(ctx, f, dir)
	case *frames.EndFrame, *frames.CancelFrame:
		s.disconnect()
		return s.PushFrame(ctx, f, dir)
	default:
		return s.PushFrame(ctx, f, dir)
	}
}

// Cleanup tears down the connection and the processor.
func (s *Service) Cleanup(ctx context.Context) error {
	s.disconnect()
	return s.Base.Cleanup(ctx)
}

func (s *Service) connect(ctx context.Context) error {
	q := url.Values{}
	q.Set("model", s.cfg.Model)
	q.Set("language", s.cfg.Language)
	q.Set("encoding", "linear16")
	q.Set("sample_rate", strconv.Itoa(s.sampleRate))
	q.Set("channels", "1")
	q.Set("interim_results", "true")
	q.Set("smart_format", "true")
	q.Set("punctuate", "true")
	q.Set("endpointing", "300")
	q.Set("utterance_end_ms", "1000")
	q.Set("vad_events", "true")

	header := http.Header{}
	header.Set("Authorization", "Token "+s.cfg.APIKey)

	conn, httpResp, err := websocket.Dial(ctx, listenURL+"?"+q.Encode(), &websocket.DialOptions{HTTPHeader: header})
	if httpResp != nil && httpResp.Body != nil {
		_ = httpResp.Body.Close()
	}
	if err != nil {
		return fmt.Errorf("deepgram dial: %w", err)
	}

	connCtx, cancel := context.WithCancel(ctx)
	s.connMu.Lock()
	s.conn = conn
	s.cancel = cancel
	s.connMu.Unlock()
	s.wg.Add(2)
	go s.readLoop(connCtx, conn)
	go s.keepAlive(connCtx)
	return nil
}

func (s *Service) disconnect() {
	s.connMu.Lock()
	cancel := s.cancel
	conn := s.conn
	if cancel == nil {
		s.connMu.Unlock()
		return
	}
	// Ask Deepgram to flush and close before tearing down.
	if conn != nil {
		_ = conn.Write(context.Background(), websocket.MessageText, []byte(`{"type":"CloseStream"}`))
	}
	s.connMu.Unlock()

	cancel()
	s.wg.Wait()

	s.connMu.Lock()
	if s.conn != nil {
		_ = s.conn.Close(websocket.StatusNormalClosure, "")
		s.conn = nil
	}
	s.cancel = nil
	s.connMu.Unlock()
}

func (s *Service) sendAudio(ctx context.Context, audio []byte) {
	s.connMu.Lock()
	defer s.connMu.Unlock()
	if s.conn == nil {
		return
	}
	_ = s.conn.Write(ctx, websocket.MessageBinary, audio)
}

func (s *Service) keepAlive(ctx context.Context) {
	defer s.wg.Done()
	ticker := time.NewTicker(keepAlivePeriod)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			s.connMu.Lock()
			if s.conn != nil {
				_ = s.conn.Write(ctx, websocket.MessageText, []byte(`{"type":"KeepAlive"}`))
			}
			s.connMu.Unlock()
		}
	}
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

func (s *Service) readLoop(ctx context.Context, conn *websocket.Conn) {
	defer s.wg.Done()
	for {
		_, data, err := conn.Read(ctx)
		if err != nil {
			return
		}
		var msg dgMessage
		if err := json.Unmarshal(data, &msg); err != nil {
			continue
		}
		if msg.Type != "Results" || len(msg.Channel.Alternatives) == 0 {
			continue
		}
		text := msg.Channel.Alternatives[0].Transcript
		if text == "" {
			continue
		}
		timestamp := time.Now().UTC().Format(time.RFC3339)
		if !msg.IsFinal {
			_ = s.PushFrame(ctx, frames.NewInterimTranscriptionFrame(text, "", timestamp), processor.Downstream)
			continue
		}
		tf := frames.NewTranscriptionFrame(text, "", timestamp)
		tf.Finalized = msg.SpeechFinal
		_ = s.PushFrame(ctx, tf, processor.Downstream)
	}
}
