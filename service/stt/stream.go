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
	"github.com/gojargo/jargo/telemetry/metrics"
	"github.com/gojargo/jargo/telemetry/tracing"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/trace"
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

// Metadata describes an STT service to downstream processors. A Connector or
// Transcriber implements Describer to provide it at pipeline start.
type Metadata struct {
	// RecommendedUserTurns is the turn strategy the service recommends; a service
	// that does its own server-side end-of-turn detection returns
	// frames.UserTurnExternal so the user aggregator can adopt external strategies.
	RecommendedUserTurns frames.UserTurnRecommendation
	// TTFSP99 is the time-to-final-segment P99 latency reported on the metadata
	// frame (see frames.STTMetadataFrame).
	TTFSP99 time.Duration
	// Model is the provider's model identifier, e.g. "nova-3". It labels the
	// metrics and is what a cost-tracking backend prices the transcription
	// against, so it should be the identifier the provider bills under.
	Model string
}

// Describer is an optional interface a Connector or Transcriber implements to
// describe its STT service. When unimplemented, the base broadcasts a plain
// STTMetadataFrame carrying only the service name.
type Describer interface {
	Metadata() Metadata
}

// StreamService is the shared processor for streaming STT providers. It manages
// the session lifecycle, forwards input audio to the Stream, and turns results
// into InterimTranscriptionFrames and TranscriptionFrames.
type StreamService struct {
	*processor.Base
	conn    Connector
	cfgRate int
	model   string

	sampleRate  int
	mu          sync.Mutex
	stream      Stream
	cancel      context.CancelFunc
	connectedAt time.Time
	audioBytes  int64
	wg          sync.WaitGroup
}

// NewStream builds a streaming STT service named name driven by conn. A non-zero
// sampleRate overrides the transport's input rate.
func NewStream(name string, conn Connector, sampleRate int) *StreamService {
	s := &StreamService{conn: conn, cfgRate: sampleRate}
	if d, ok := conn.(Describer); ok {
		s.model = d.Metadata().Model
	}
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
		s.broadcastMetadata(ctx)
		s.sampleRate = s.cfgRate
		if s.sampleRate == 0 {
			s.sampleRate = fr.AudioInSampleRate
		}
		return s.connect(ctx)
	case *frames.InputAudioRawFrame:
		s.send(fr.Audio)
		return s.PushFrame(ctx, f, dir)
	case *frames.EndFrame, *frames.CancelFrame:
		s.disconnect(ctx)
		return s.PushFrame(ctx, f, dir)
	default:
		return s.PushFrame(ctx, f, dir)
	}
}

// Cleanup tears down the session and the processor.
func (s *StreamService) Cleanup(ctx context.Context) error {
	s.disconnect(ctx)
	return s.Base.Cleanup(ctx)
}

// broadcastMetadata pushes the STT service's metadata frame downstream at
// pipeline start, enriched from the Connector when it implements Describer.
func (s *StreamService) broadcastMetadata(ctx context.Context) {
	mf := frames.NewSTTMetadataFrame(0)
	mf.ServiceName = s.Name()
	if d, ok := s.conn.(Describer); ok {
		m := d.Metadata()
		mf.UserTurns = m.RecommendedUserTurns
		mf.TTFSP99Latency = m.TTFSP99
	}
	_ = s.PushFrame(ctx, mf, processor.Downstream)
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
	s.connectedAt = time.Now()
	s.audioBytes = 0
	s.mu.Unlock()
	s.wg.Go(func() { s.readLoop(connCtx, stream) })
	return nil
}

// disconnect closes the session and records what it transcribed. Streaming
// providers bill for every second of audio the session carries, silence
// included, so the usage is the audio streamed over the whole session rather
// than per transcript — and it is reported on one span covering the session,
// started at the point the connection opened.
func (s *StreamService) disconnect(ctx context.Context) {
	s.mu.Lock()
	cancel := s.cancel
	stream := s.stream
	connectedAt := s.connectedAt
	audioBytes := s.audioBytes
	s.cancel = nil
	s.stream = nil
	s.connectedAt = time.Time{}
	s.mu.Unlock()
	if cancel == nil {
		return
	}
	cancel()
	s.wg.Wait()
	_ = stream.Close()
	s.recordUsage(ctx, connectedAt, audioBytes)
}

// recordUsage emits the session's STT span, spanning the life of the connection.
func (s *StreamService) recordUsage(ctx context.Context, connectedAt time.Time, audioBytes int64) {
	audio := pcmDuration(audioBytes, s.sampleRate)
	if audio == 0 {
		return
	}
	metrics.RecordSTTAudio(ctx, s.Name(), s.model, audio.Seconds())
	sctx, span := tracing.Tracer().Start(ctx, "stt", trace.WithTimestamp(connectedAt))
	defer span.End()
	span.SetAttributes(
		attribute.String("stt.service", s.Name()),
		attribute.Int("stt.sample_rate", s.sampleRate),
		attribute.Int64("stt.audio_ms", audio.Milliseconds()),
	)
	tracing.SetSTTUsage(sctx, s.model, audio)
}

func (s *StreamService) send(audio []byte) {
	s.mu.Lock()
	stream := s.stream
	if stream != nil {
		s.audioBytes += int64(len(audio))
	}
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

// pcmDuration is how long n bytes of 16-bit mono PCM play for at sampleRate.
func pcmDuration(n int64, sampleRate int) time.Duration {
	if n <= 0 || sampleRate <= 0 {
		return 0
	}
	return time.Duration(n) * time.Second / time.Duration(2*sampleRate)
}
