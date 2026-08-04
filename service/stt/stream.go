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
	"github.com/gojargo/jargo/language"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/service/settings"
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

// Finalizer is an optional interface a Stream implements when the provider has
// to be told the speech has ended before it flushes the transcript for it.
// Finalize is called once the VAD reports the user stopped, and the session
// carries on afterwards for the next utterance. A Stream that does not implement
// it is left to the provider's own endpointing.
type Finalizer interface {
	Finalize() error
}

// SettingsHolder is an optional interface a Connector implements when part of
// what it was built with can change while the pipeline runs: the language it
// transcribes, the model it uses. The value returned is the provider's own
// store, a pointer to a settings value, which an update is merged into.
type SettingsHolder interface {
	Settings() any
}

// SettingsUpdater is an optional interface a Connector implements to act on a
// settings change. A Connector that holds settings without implementing this
// still has them updated; it simply picks them up the next time it opens a
// session.
type SettingsUpdater interface {
	SettingsHolder
	// UpdateSettings is called once the changed fields have been written to the
	// store, with what changed and what each field held before. Returning true
	// asks for the session to be reopened, which is what a provider needs when
	// the setting is fixed at the point the session opens and cannot be changed
	// on a session already running.
	UpdateSettings(ctx context.Context, changed settings.Changed) (reopen bool, err error)
}

// LanguageNamer is an optional interface a Connector implements to name a
// language the way its provider does. Providers disagree on the codes, so a
// caller naming a language neutrally has it converted before it is stored,
// leaving the store holding the code the provider itself uses. Without that the
// stored value and the next update would be in different vocabularies, and a
// change would be reported for two spellings of the same language.
type LanguageNamer interface {
	ServiceLanguage(l language.Language) string
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

	// Reopening the session for a settings change. Doing it while the user is
	// mid-sentence would drop what they are saying, so it waits for the turn to
	// end, and the audio that arrives in the gap is held rather than sent to a
	// session that is being replaced.
	//
	// canReopen is false from the moment speech is detected until the turn ends.
	canReopen bool
	// needReopen records a reopen asked for while the user was speaking.
	needReopen bool
	// reopening is set while the session is being replaced, which is what says
	// to hold incoming audio rather than send it.
	reopening bool
	// held is the audio that arrived while the session was being replaced, sent
	// on once the new one is up so the words spoken across the gap are not lost.
	held [][]byte

	// settingsMu serializes reading the provider's settings against changing
	// them. A session is opened from the read loop when one drops, and an update
	// is applied on the frame goroutine, so without this a session could be
	// opened from settings that are half written.
	settingsMu sync.Mutex
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
	s.canReopen = true
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
	case *frames.STTUpdateSettingsFrame:
		if !fr.TargetsService(s) {
			// Meant for another service; leave it untouched for that one.
			return s.PushFrame(ctx, f, dir)
		}
		s.updateSettings(ctx, fr)
		return nil
	case *frames.VADUserStartedSpeakingFrame:
		s.mu.Lock()
		s.canReopen = false
		s.mu.Unlock()
		return s.PushFrame(ctx, f, dir)
	case *frames.VADUserStoppedSpeakingFrame:
		s.finalize()
		return s.PushFrame(ctx, f, dir)
	case *frames.UserStoppedSpeakingFrame:
		s.reopenIfDeferred(ctx)
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
	// Held across the dial: the provider reads its settings to build the
	// session, and a session opened from settings changing underneath it would
	// be neither the old configuration nor the new one.
	s.settingsMu.Lock()
	stream, err := s.conn.Connect(sessionCtx, s.sampleRate)
	s.settingsMu.Unlock()
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
	s.mu.Lock()
	model := s.model
	s.mu.Unlock()
	metrics.RecordSTTAudio(ctx, s.Name(), model, audio.Seconds())
	s.pushUsageMetrics(ctx, audio)
	sctx, span := tracing.Tracer().Start(
		s.Tracing().Parent(ctx), "stt", trace.WithTimestamp(connectedAt))
	defer span.End()
	span.SetAttributes(
		attribute.String("stt.service", s.Name()),
		attribute.Int("stt.sample_rate", s.sampleRate),
		attribute.Int64("stt.audio_ms", audio.Milliseconds()),
	)
	tracing.SetSTTUsage(sctx, model, audio)
}

// updateSettings merges an update into the provider's own settings and lets it
// act on what changed, reopening the session when the provider says the change
// cannot take effect on the one already running.
func (s *StreamService) updateSettings(ctx context.Context, f *frames.STTUpdateSettingsFrame) {
	holder, ok := s.conn.(SettingsHolder)
	if !ok {
		slog.Warn("settings update for a service whose provider has none", "service", s.Name())
		return
	}
	store := holder.Settings()

	// The reopen this may ask for is taken after the lock is released, or it
	// would deadlock against the dial it goes on to make.
	s.settingsMu.Lock()
	reopen, err := s.applySettings(ctx, store, f)
	s.settingsMu.Unlock()
	if err != nil {
		s.PushError(ctx, "stt: settings update", err, false)
		return
	}
	if reopen {
		s.requestReopen(ctx)
	}
}

// applySettings merges the update and lets the provider act on it, reporting
// whether the provider wants the session reopened. It runs under settingsMu.
func (s *StreamService) applySettings(
	ctx context.Context, store any, f *frames.STTUpdateSettingsFrame,
) (bool, error) {
	delta, ok, err := settings.Resolve(&f.ServiceUpdateSettingsFrame, store)
	if err != nil {
		return false, err
	}
	if !ok {
		return false, nil
	}

	// Naming the language the provider's way before applying is what keeps the
	// comparison honest: the store holds the provider's code, so a neutral name
	// that means the same language must be converted first or it reads as a
	// change when nothing changed.
	s.nameLanguage(delta)

	changed, err := settings.Apply(store, delta)
	if err != nil {
		return false, err
	}
	if len(changed) == 0 {
		return false, nil
	}
	slog.Info("updated settings", "service", s.Name(), "fields", changed.String())

	if changed.Has("model") {
		// The model labels the usage this service reports, and it is priced
		// against it, so a model that changed mid-call has to relabel what
		// follows or the cost lands against the wrong one.
		s.syncModel(store)
	}

	updater, ok := s.conn.(SettingsUpdater)
	if !ok {
		return false, nil
	}
	return updater.UpdateSettings(ctx, changed)
}

// nameLanguage rewrites a language the delta gives into the code the provider
// uses, when the provider says how. A code the provider does not recognize is
// left as it came, since it may be one the service accepts directly.
func (s *StreamService) nameLanguage(delta any) {
	namer, ok := s.conn.(LanguageNamer)
	if !ok {
		return
	}
	value, ok := settings.Get(delta, "language")
	if !ok {
		return
	}
	code, ok := value.(string)
	if !ok || code == "" {
		return
	}
	named := namer.ServiceLanguage(language.Language(code))
	if named == "" || named == code {
		return
	}
	if err := settings.SetNamed(delta, "language", named); err != nil {
		slog.Warn("naming the language the provider's way failed",
			"service", s.Name(), "err", err)
	}
}

// syncModel relabels the metrics with the model now in force.
func (s *StreamService) syncModel(store any) {
	name, _ := settings.Get(store, "model")
	model, _ := name.(string)
	s.mu.Lock()
	s.model = model
	s.mu.Unlock()
}

// requestReopen replaces the session now, or as soon as the user has finished
// speaking. Reopening mid-sentence would drop what is being said, and the words
// lost are exactly the ones the change was meant to transcribe better.
func (s *StreamService) requestReopen(ctx context.Context) {
	s.mu.Lock()
	can := s.canReopen
	if !can {
		s.needReopen = true
	}
	s.mu.Unlock()
	if can {
		s.reopen(ctx)
	}
}

// reopenIfDeferred runs a reopen that waited for the user to stop speaking.
func (s *StreamService) reopenIfDeferred(ctx context.Context) {
	s.mu.Lock()
	s.canReopen = true
	deferred := s.needReopen
	s.mu.Unlock()
	if deferred {
		s.reopen(ctx)
	}
}

// reopen replaces the session with one carrying the new settings, holding the
// audio that arrives in between and sending it on once the new session is up, so
// the words spoken across the gap still reach the provider.
func (s *StreamService) reopen(ctx context.Context) {
	s.mu.Lock()
	s.held = nil
	s.reopening = true
	s.needReopen = false
	s.mu.Unlock()

	slog.Info("reopening the transcription session", "service", s.Name())
	s.disconnect(ctx)
	err := s.connect(ctx)

	s.mu.Lock()
	s.reopening = false
	held := s.held
	s.held = nil
	s.mu.Unlock()

	if err != nil {
		s.PushError(ctx, "stt: reopening the session", err, false)
		return
	}
	for _, audio := range held {
		s.send(audio)
	}
}

func (s *StreamService) send(audio []byte) {
	s.mu.Lock()
	if s.reopening {
		// The session is being replaced. Hold the audio rather than send it to
		// one that is going away, or the words spoken across the gap are lost.
		s.held = append(s.held, audio)
		s.mu.Unlock()
		return
	}
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

// finalize tells the session the speech has ended, for a provider that flushes
// the transcript for an utterance only when asked. It is a no-op for a provider
// that does its own endpointing, and for a session that is being replaced: what
// the finalize would flush is held audio the new session has not been given yet.
func (s *StreamService) finalize() {
	s.mu.Lock()
	stream := s.stream
	reopening := s.reopening
	s.mu.Unlock()
	if stream == nil || reopening {
		return
	}
	fin, ok := stream.(Finalizer)
	if !ok {
		return
	}
	if err := fin.Finalize(); err != nil {
		slog.Debug("stt finalize failed", "service", s.Name(), "err", err)
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
