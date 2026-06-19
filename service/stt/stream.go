// Package stt is the shared base for speech-to-text services. It offers two
// shapes: StreamService for providers that transcribe continuously over a live
// connection (the result is a stream of interim and final transcriptions), and
// SegmentService for batch providers that transcribe a whole utterance at once
// (delimited by a turn detector upstream).
//
// A concrete provider supplies only the part that differs — a Connector that
// opens a session, or a Transcriber that transcribes one segment — while the
// frame contract and lifecycle live here.
package stt

import (
	"context"
	"sync"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor"
)

// Result is one transcription result from a streaming STT provider.
type Result struct {
	// Text is the transcribed text.
	Text string
	// Final reports whether this is a finalized result rather than an interim
	// partial.
	Final bool
	// EndOfTurn reports whether this result marks the end of the user's turn,
	// the point at which the aggregator may trigger the LLM.
	EndOfTurn bool
	// Language is the detected language as a BCP-47 tag, or "" when unknown.
	Language string
}

// Stream is a live STT session opened by a Connector. The service writes audio
// with Send and reads results with Recv until Recv returns an error (including
// io.EOF), then calls Close.
type Stream interface {
	// Send writes a chunk of 16-bit mono PCM audio to the session.
	Send(audio []byte) error
	// Recv blocks for the next batch of results, returning an error when the
	// session ends.
	Recv() ([]Result, error)
	// Close tears the session down.
	Close() error
}

// Connector opens a streaming STT session for the given input sample rate.
type Connector interface {
	Connect(ctx context.Context, sampleRate int) (Stream, error)
}

// StreamService is the shared processor for streaming STT providers. It manages
// the session lifecycle, forwards input audio to the Stream, and turns results
// into InterimTranscriptionFrames and TranscriptionFrames.
type StreamService struct {
	*processor.Base
	conn    Connector
	cfgRate int

	sampleRate int
	mu         sync.Mutex
	stream     Stream
	cancel     context.CancelFunc
	wg         sync.WaitGroup
}

// NewStream builds a streaming STT service named name driven by conn. A non-zero
// sampleRate overrides the transport's input rate.
func NewStream(name string, conn Connector, sampleRate int) *StreamService {
	s := &StreamService{conn: conn, cfgRate: sampleRate}
	s.Base = processor.New(name, s)
	return s
}

// ProcessFrame manages the connection lifecycle and streams audio.
func (s *StreamService) ProcessFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	if err := s.Base.ProcessFrame(ctx, f, dir); err != nil {
		return err
	}
	switch fr := f.(type) {
	case *frames.StartFrame:
		if err := s.PushFrame(ctx, f, dir); err != nil {
			return err
		}
		s.sampleRate = s.cfgRate
		if s.sampleRate == 0 {
			s.sampleRate = fr.AudioInSampleRate
		}
		return s.connect(ctx)
	case *frames.InputAudioRawFrame:
		s.send(fr.Audio)
		return s.PushFrame(ctx, f, dir)
	case *frames.EndFrame, *frames.CancelFrame:
		s.disconnect()
		return s.PushFrame(ctx, f, dir)
	default:
		return s.PushFrame(ctx, f, dir)
	}
}

// Cleanup tears down the session and the processor.
func (s *StreamService) Cleanup(ctx context.Context) error {
	s.disconnect()
	return s.Base.Cleanup(ctx)
}

func (s *StreamService) connect(ctx context.Context) error {
	connCtx, cancel := context.WithCancel(ctx)
	stream, err := s.conn.Connect(connCtx, s.sampleRate)
	if err != nil {
		cancel()
		return err
	}
	s.mu.Lock()
	s.stream = stream
	s.cancel = cancel
	s.mu.Unlock()
	s.wg.Go(func() { s.readLoop(connCtx, stream) })
	return nil
}

func (s *StreamService) disconnect() {
	s.mu.Lock()
	cancel := s.cancel
	stream := s.stream
	s.cancel = nil
	s.stream = nil
	s.mu.Unlock()
	if cancel == nil {
		return
	}
	cancel()
	s.wg.Wait()
	_ = stream.Close()
}

func (s *StreamService) send(audio []byte) {
	s.mu.Lock()
	stream := s.stream
	s.mu.Unlock()
	if stream != nil {
		_ = stream.Send(audio)
	}
}

func (s *StreamService) readLoop(ctx context.Context, stream Stream) {
	for {
		results, err := stream.Recv()
		if err != nil {
			return
		}
		for _, r := range results {
			s.emit(ctx, r)
		}
	}
}

func (s *StreamService) emit(ctx context.Context, r Result) {
	if r.Text == "" {
		return
	}
	ts := time.Now().UTC().Format(time.RFC3339)
	if !r.Final {
		f := frames.NewInterimTranscriptionFrame(r.Text, "", ts)
		f.Language = r.Language
		_ = s.PushFrame(ctx, f, processor.Downstream)
		return
	}
	f := frames.NewTranscriptionFrame(r.Text, "", ts)
	f.Finalized = r.EndOfTurn
	f.Language = r.Language
	_ = s.PushFrame(ctx, f, processor.Downstream)
}
