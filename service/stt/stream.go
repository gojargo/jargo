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
	"errors"
	"log/slog"
	"sync"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/service/wsservice"
	"github.com/gojargo/jargo/telemetry/metrics"
	"github.com/gojargo/jargo/telemetry/tracing"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/trace"
)

// errNoSession is returned when a Connector reports neither a session nor an
// error, which leaves nothing to transcribe on.
//
//nolint:gochecknoglobals // sentinel error
var errNoSession = errors.New("stt: connector returned no session")

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

const (
	// keepaliveSilence is how much silence one keepalive submits.
	keepaliveSilence = 100 * time.Millisecond
	// DefaultKeepaliveInterval is how often the idle check runs when a provider
	// asks for a keepalive without saying how often to look.
	DefaultKeepaliveInterval = 5 * time.Second
)

// KeepaliveOptions says how a provider wants an idle session held open.
type KeepaliveOptions struct {
	// Timeout is how long a session may carry no audio before silence is sent
	// to keep it open. Zero disables the keepalive, which is right for a
	// provider that leaves an idle session alone.
	Timeout time.Duration
	// Interval is how often the session is checked for having gone idle. Zero
	// takes DefaultKeepaliveInterval.
	Interval time.Duration
}

// Keepaliver is an optional interface a Connector implements when the provider
// closes a session that carries no audio for a while. Silence only reaches the
// service during a call when nobody is speaking, and a service switched out of
// the pipeline sends nothing at all, so an idle session is a normal thing to
// have and a closed one costs the next thing the user says. A Connector that
// does not implement this gets no keepalive, which is right for a provider that
// leaves an idle session open.
type Keepaliver interface {
	Keepalive() KeepaliveOptions
}

// KeepaliveSender is an optional interface a Stream implements when the silence
// has to be wrapped in the provider's own protocol, or replaced by a protocol
// message of its own, rather than submitted as raw audio. A Stream that does not
// implement it has the silence sent with Send, and that silence is audio the
// provider bills for, so it counts towards the session's usage. Silence a
// provider swaps for a protocol message is not audio and is not counted.
type KeepaliveSender interface {
	SendKeepalive(silence []byte) error
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
//
// The session is held open for the length of the call, so it can drop while the
// call carries on. The service reopens it rather than transcribing nothing for
// the rest of the call: see service/wsservice, which runs the read loop and the
// reconnection around it. A dropped session costs the audio spoken during the
// gap, which is why the reconnect is immediate rather than waiting for the next
// utterance.
type StreamService struct {
	*processor.Base
	conn    Connector
	cfgRate int
	model   string
	ws      *wsservice.Base

	sampleRate int
	mu         sync.Mutex
	stream     Stream
	// sessionCancel ends the current session, replaced on every reconnect.
	sessionCancel context.CancelFunc
	// readCancel ends the read loop, and with it the whole connection. It
	// outlives any one session, so a reconnect does not stop the loop.
	readCancel  context.CancelFunc
	connectedAt time.Time
	audioBytes  int64
	wg          sync.WaitGroup

	// keepalive is what the provider asked for; a zero Timeout means none.
	keepalive KeepaliveOptions
	// lastAudio is when audio last went to the provider, which is what says
	// whether the session has gone idle.
	lastAudio time.Time
	// keepaliveCancel stops the keepalive goroutine, nil when none is running.
	keepaliveCancel context.CancelFunc
	keepaliveWG     sync.WaitGroup
}

// NewStream builds a streaming STT service named name driven by conn. A non-zero
// sampleRate overrides the transport's input rate.
func NewStream(name string, conn Connector, sampleRate int) *StreamService {
	s := &StreamService{conn: conn, cfgRate: sampleRate}
	if d, ok := conn.(Describer); ok {
		s.model = d.Metadata().Model
	}
	if k, ok := conn.(Keepaliver); ok {
		s.keepalive = k.Keepalive()
		if s.keepalive.Interval <= 0 {
			s.keepalive.Interval = DefaultKeepaliveInterval
		}
	}
	s.Base = processor.New(name, s)
	s.ws = wsservice.New(s, wsservice.Config{})
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

// connect opens the session and starts the read loop that owns it from here on.
// The loop outlives any one session: when the session drops it opens another,
// which is why the loop context is separate from the session's.
func (s *StreamService) connect(ctx context.Context) error {
	readCtx, cancel := context.WithCancel(ctx)
	s.ws.Connect()
	if err := s.ConnectWebsocket(readCtx); err != nil {
		cancel()
		return err
	}
	s.mu.Lock()
	s.readCancel = cancel
	s.connectedAt = time.Now()
	s.audioBytes = 0
	s.mu.Unlock()
	s.wg.Go(func() { s.ws.ReceiveTaskHandler(readCtx, s.reportConnectionError) })
	return nil
}

// disconnect closes the session and records what it transcribed. Streaming
// providers bill for every second of audio the session carries, silence
// included, so the usage is the audio streamed over the whole connection rather
// than per transcript — and it is reported on one span covering it, started at
// the point the connection opened. A session dropped and reopened partway
// through is still the same connection, so it is still one span.
func (s *StreamService) disconnect(ctx context.Context) {
	// Marking the disconnect deliberate first is what tells the read loop that
	// the session ending is the shutdown it asked for, not one to reconnect.
	s.ws.Disconnect()
	s.mu.Lock()
	cancel := s.readCancel
	connectedAt := s.connectedAt
	audioBytes := s.audioBytes
	s.readCancel = nil
	s.connectedAt = time.Time{}
	s.mu.Unlock()
	if cancel == nil {
		return
	}
	cancel()
	s.wg.Wait()
	_ = s.DisconnectWebsocket(ctx)
	s.recordUsage(ctx, connectedAt, audioBytes)
}

// ConnectWebsocket opens a transcription session. The read loop calls it for
// every reconnect, so it replaces whatever session was there before.
func (s *StreamService) ConnectWebsocket(ctx context.Context) error {
	sessionCtx, cancel := context.WithCancel(ctx)
	stream, err := s.conn.Connect(sessionCtx, s.sampleRate)
	if err != nil {
		cancel()
		return err
	}
	if stream == nil {
		cancel()
		return errNoSession
	}
	s.mu.Lock()
	s.stream = stream
	s.sessionCancel = cancel
	s.mu.Unlock()
	// Every session gets its own keepalive, the first and each replacement
	// alike: the goroutine gives up on a failed send, so the one belonging to
	// the session that just dropped is likely gone.
	s.startKeepalive(ctx)
	return nil
}

// DisconnectWebsocket closes the current transcription session, if there is one.
// A session that has already dropped cannot be closed cleanly, and that says
// nothing about whether the next one will open, so the failure is logged rather
// than failing the reconnect attempt that is about to redial.
func (s *StreamService) DisconnectWebsocket(context.Context) error {
	// Stopped before the session goes, so nothing tries to send on a session
	// that is being closed underneath it.
	s.stopKeepalive()
	s.mu.Lock()
	stream := s.stream
	cancel := s.sessionCancel
	s.stream = nil
	s.sessionCancel = nil
	s.mu.Unlock()
	if cancel != nil {
		cancel()
	}
	if stream == nil {
		return nil
	}
	if err := stream.Close(); err != nil {
		slog.Debug("closing the transcription session failed", "service", s.Name(), "err", err)
	}
	return nil
}

// ReceiveMessages reads results until the session ends, pushing each one
// downstream. It returns the failure that ended the session, leaving the read
// loop to decide whether to open another.
func (s *StreamService) ReceiveMessages(ctx context.Context) error {
	s.mu.Lock()
	stream := s.stream
	s.mu.Unlock()
	if stream == nil {
		return errNoSession
	}
	for {
		results, err := stream.Recv()
		if err != nil {
			return err
		}
		for _, r := range results {
			s.emit(ctx, r)
		}
	}
}

// Connected reports whether there is a session to transcribe on.
func (s *StreamService) Connected() bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.stream != nil
}

// reportConnectionError puts a lost provider connection on the pipeline. It is
// not fatal: the call continues, and the application decides what losing
// transcription for part of it means.
func (s *StreamService) reportConnectionError(ctx context.Context, message string) {
	s.PushError(ctx, message, nil, false)
}

// recordUsage emits the session's STT span, spanning the life of the connection.
func (s *StreamService) recordUsage(ctx context.Context, connectedAt time.Time, audioBytes int64) {
	audio := pcmDuration(audioBytes, s.sampleRate)
	if audio == 0 {
		return
	}
	metrics.RecordSTTAudio(ctx, s.Name(), s.model, audio.Seconds())
	s.pushUsageMetrics(ctx, audio)
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
	// Stamped whether or not there is a session to send on: audio arriving is
	// what says the call is not idle, and a session that opens a moment later
	// has no catching up to do.
	s.lastAudio = time.Now()
	if stream != nil {
		s.audioBytes += int64(len(audio))
	}
	s.mu.Unlock()
	if stream != nil {
		_ = stream.Send(audio)
	}
}

// startKeepalive replaces the running keepalive, if any, with one for the
// session just opened. It does nothing when the provider asked for none.
func (s *StreamService) startKeepalive(ctx context.Context) {
	if s.keepalive.Timeout <= 0 {
		return
	}
	s.stopKeepalive()
	kaCtx, cancel := context.WithCancel(ctx)
	s.mu.Lock()
	s.keepaliveCancel = cancel
	s.lastAudio = time.Now()
	s.mu.Unlock()
	s.keepaliveWG.Go(func() { s.keepaliveLoop(kaCtx) })
}

// stopKeepalive ends the running keepalive and waits for it to finish.
func (s *StreamService) stopKeepalive() {
	s.mu.Lock()
	cancel := s.keepaliveCancel
	s.keepaliveCancel = nil
	s.mu.Unlock()
	if cancel == nil {
		return
	}
	cancel()
	s.keepaliveWG.Wait()
}

// keepaliveLoop submits silence to a session that has carried no audio for long
// enough that the provider might close it. It gives up on the first failed send:
// the session is gone, and the read loop is the one that reopens it.
func (s *StreamService) keepaliveLoop(ctx context.Context) {
	ticker := time.NewTicker(s.keepalive.Interval)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
		}
		if !s.Connected() {
			continue
		}
		s.mu.Lock()
		idle := time.Since(s.lastAudio)
		s.mu.Unlock()
		if idle < s.keepalive.Timeout {
			continue
		}
		if err := s.sendKeepalive(); err != nil {
			slog.Warn("keeping the transcription session alive failed",
				"service", s.Name(), "err", err)
			return
		}
		s.mu.Lock()
		s.lastAudio = time.Now()
		s.mu.Unlock()
	}
}

// sendKeepalive submits one stretch of silence on the current session.
func (s *StreamService) sendKeepalive() error {
	s.mu.Lock()
	stream := s.stream
	rate := s.sampleRate
	s.mu.Unlock()
	if stream == nil {
		return errNoSession
	}

	// 16-bit mono silence, which is what every provider is fed.
	silence := make([]byte, int(float64(rate)*keepaliveSilence.Seconds())*2)

	if ks, ok := stream.(KeepaliveSender); ok {
		// The provider swapped the silence for something of its own, so there
		// is no audio to bill for.
		return ks.SendKeepalive(silence)
	}
	if err := stream.Send(silence); err != nil {
		return err
	}
	// Silence submitted as audio is audio the provider bills for.
	s.mu.Lock()
	s.audioBytes += int64(len(silence))
	s.mu.Unlock()
	return nil
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

// CanGenerateMetrics reports that this service times transcription and reports
// the result, so the pipeline counts it when it collects the processors that
// report metrics.
func (s *StreamService) CanGenerateMetrics() bool { return true }
