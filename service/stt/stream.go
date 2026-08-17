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
	"github.com/gojargo/jargo/service"
	"github.com/gojargo/jargo/service/settings"
	"github.com/gojargo/jargo/service/wsservice"
	"github.com/gojargo/jargo/telemetry/metrics"
	"github.com/gojargo/jargo/telemetry/tracing"
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
	// Speech reports a speech boundary the provider detected itself. Most
	// providers leave it unset and let the pipeline's own detection decide when
	// the user is speaking; a provider running detection server-side sets it so
	// the pipeline can defer to that instead. A result may carry a boundary and
	// no text.
	Speech SpeechState
	// Interrupt asks for the bot to be interrupted along with a SpeechStarted
	// boundary, which is what a barge-in on the provider's own detection means.
	// It is ignored on any other boundary.
	Interrupt bool
	// FromFinalize reports that this result is the provider answering the
	// finalize it was asked for. Only a provider that confirms a flush sets it,
	// and only then is the transcript that follows the last one for the
	// utterance: a result may be final without being the end of the turn. A
	// provider that flushes without confirming leaves it unset and says what it
	// knows through EndOfTurn instead.
	FromFinalize bool
}

// SpeechState is a speech boundary a provider detected on its own.
type SpeechState int

const (
	// SpeechUnknown means the result says nothing about speech boundaries,
	// which is the case for every provider that leaves detection to the
	// pipeline.
	SpeechUnknown SpeechState = iota
	// SpeechStarted means the provider heard the user begin speaking.
	SpeechStarted
	// SpeechStopped means the provider heard the user stop.
	SpeechStopped
)

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

// SpeechStarter is an optional interface a Stream implements when the start of
// an utterance is something the session itself has to know about. SpeechStarted
// is called once the VAD reports the user began speaking. It is where a session
// that builds a transcript across several results drops what it was holding: the
// utterance before it is over whether or not the provider ever closed it, and
// text left over from it would otherwise be read as the beginning of this one. A
// Stream that keeps nothing between utterances does not implement it.
type SpeechStarter interface {
	SpeechStarted()
}

// SettingsHolder is an optional interface a provider implements when part of
// what it was built with can change while the pipeline runs: the language it
// transcribes, the model it uses. The value returned is the provider's own
// store, a pointer to a settings value, which an update is merged into.
//
// Either kind of provider may implement it, a Connector or a Transcriber, and
// either kind of service applies an update the same way.
type SettingsHolder interface {
	Settings() any
}

// SettingsUpdater is an optional interface a provider implements to act on a
// settings change. A provider that holds settings without implementing this
// still has them updated; it simply picks them up the next time it is used.
type SettingsUpdater interface {
	SettingsHolder
	// UpdateSettings is called once the changed fields have been written to the
	// store, with what changed and what each field held before. Returning true
	// asks for the session to be reopened, which is what a provider needs when
	// the setting is fixed at the point the session opens and cannot be changed
	// on a session already running. A segmented service has no session to
	// reopen and transcribes the next segment with the settings as they stand,
	// so it ignores the request.
	UpdateSettings(ctx context.Context, changed settings.Changed) (reopen bool, err error)
}

// LanguageNamer is an optional interface a provider implements to name a
// language the way that provider does. Providers disagree on the codes, so a
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
	// frame (see frames.STTMetadataFrame). Left unset, the service is described
	// with DefaultTTFSP99.
	TTFSP99 time.Duration
	// SupportsTTFS reports whether TTFS means anything for this service; nil
	// defaults to true. A service whose server defines the turn boundary sets it
	// false, and is described with no wait at all rather than a measured one.
	SupportsTTFS *bool
	// Model is the provider's model identifier, e.g. "nova-3". It labels the
	// metrics and is what a cost-tracking backend prices the transcription
	// against, so it should be the identifier the provider bills under.
	Model string
}

// ttfs is the latency a described service is published with: none when TTFS
// does not apply to it, its own measurement when it carries one, and the
// fallback when it does not.
func (m Metadata) ttfs(service string) time.Duration {
	if m.SupportsTTFS != nil && !*m.SupportsTTFS {
		return 0
	}
	if m.TTFSP99 > 0 {
		return m.TTFSP99
	}
	slog.Warn("stt: no TTFS p99 latency measured for this service, using the fallback",
		"service", service, "fallback", DefaultTTFSP99)
	return DefaultTTFSP99
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
	*service.Base
	conn    Connector
	cfgRate int
	model   string
	ws      *wsservice.Base
	ttfb    *ttfbTracker
	work    *processingMeter
	tracer  *segmentTracer

	sampleRate int
	mu         sync.Mutex
	stream     Stream
	// sendFailed records that the session refused the audio it was last given,
	// which takes it out of use until the read loop replaces it. Writing on
	// indefinitely is how a call goes quiet with nothing in the log to say
	// whether the provider heard nothing or the audio never reached it, and
	// repeating the send would repeat the warning for every 20ms of speech.
	// What the session no longer takes is still audio the call submitted, so it
	// still counts towards usage.
	sendFailed bool
	// sessionCancel ends the current session, replaced on every reconnect.
	sessionCancel context.CancelFunc
	// readCancel ends the read loop, and with it the whole connection. It
	// outlives any one session, so a reconnect does not stop the loop.
	readCancel  context.CancelFunc
	connectedAt time.Time
	audioBytes  int64
	// speaking tracks the boundary a provider that runs its own detection last
	// reported, so the frames go out in pairs however the provider repeats
	// itself.
	speaking bool
	wg       sync.WaitGroup

	// The finalize asked of the provider and the answer that came back. A
	// provider that confirms the flush it was asked for is the only one that
	// says which transcript closes the utterance, so the request is recorded and
	// matched against the confirmation rather than read off any one result.
	//
	// finalizeRequested is set when the provider is told the speech ended, and
	// stands unanswered for a provider that does not confirm.
	finalizeRequested bool
	// finalizePending is set once the provider confirms, and marks the next
	// transcript as the last one for the utterance.
	finalizePending bool

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
	// set applies settings updates to the provider's own store.
	set *providerSettings
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
	s.Base = service.New(name, s)
	s.set = &providerSettings{provider: conn, name: s.Name, onModel: s.setModel, onChanged: s.SettingsUpdated}
	s.ws = wsservice.New(s, wsservice.Config{})
	s.ttfb = newTTFBTracker(s.Base.Base, s.modelName)
	s.work = newProcessingMeter(s.Base.Base, s.modelName)
	s.tracer = newSegmentTracer(s.Base.Base, func() tracing.STTAttributes {
		return tracing.STTAttributes{
			Service:  s.TypeName(),
			Model:    s.modelName(),
			Settings: s.traceSettings(),
			// A streaming service is sent audio continuously and is told where
			// the speech is by the voice activity detector upstream of it.
			VADEnabled: true,
		}
	})
	s.ttfb.onReport = func(d time.Duration, end time.Time) {
		s.tracer.recordTTFB(d)
		// A wait that ended without the transcript it was for leaves a segment
		// nothing will close, so it is closed here at the moment measured to.
		s.tracer.abandon(end)
	}
	return s
}

// STTService marks this processor as a speech-to-text service.
//
// It says what a type assertion cannot: that the processor transcribes speech
// rather than merely handling transcripts. An observer outside this package
// asserts for it to tell a transcript a transcriber produced from one that
// arrived some other way.
func (s *StreamService) STTService() {}

// traceSettings renders the provider's settings for a transcription span. A
// provider that keeps no settings of its own contributes none.
func (s *StreamService) traceSettings() map[string]any { return s.set.traceSettings() }

// setModel relabels what this service reports with the model now in force.
func (s *StreamService) setModel(model string) {
	s.mu.Lock()
	s.model = model
	s.mu.Unlock()
}

// SetTTFBTimeout sets how long the service waits after the speech ends for the
// transcript that closes it, before reporting the latency against whatever
// arrived in the meantime. Zero restores DefaultTTFBTimeout. Raise it for a
// provider whose final transcript regularly takes longer than the default, which
// would otherwise be timed against an interim.
func (s *StreamService) SetTTFBTimeout(d time.Duration) { s.ttfb.setTimeout(d) }

// modelName is the model the transcription is measured and priced against.
func (s *StreamService) modelName() string {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.model
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
		s.sendAudio(fr.Audio)
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
		// A finalize left over from the utterance before it would mark the
		// first transcript of this one as the last, so the pair starts clean.
		s.finalizeRequested = false
		s.finalizePending = false
		s.mu.Unlock()
		s.speechStarted()
		s.ttfb.speechStarted()
		s.tracer.speechStarted(fr.SpeechStart())
		// The service is at work on this utterance from here until it produces
		// the transcript for it.
		s.work.begin()
		return s.PushFrame(ctx, f, dir)
	case *frames.VADUserStoppedSpeakingFrame:
		s.ttfb.speechEnded(ctx, fr)
		s.finalize()
		return s.PushFrame(ctx, f, dir)
	case *frames.InterruptionFrame:
		// The utterance being measured is not the one that matters any more.
		s.ttfb.interrupted()
		return s.PushFrame(ctx, f, dir)
	case *frames.UserStoppedSpeakingFrame:
		if err := s.PushFrame(ctx, f, dir); err != nil {
			return err
		}
		// A provider that never marks a transcript as the last one leaves the
		// segment open. The user turn is over, so whatever it heard is what it
		// heard, and the segment is closed as one that never finalized.
		s.tracer.abandon(time.Time{})
		s.reopenIfDeferred(ctx)
		return nil
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
	s.ttfb.close()
	return s.Base.Cleanup(ctx)
}

// PushFrame pushes a frame on, timing the transcripts on their way out. A
// transcript answering a finalize the provider confirmed is marked as the one
// that closes the utterance, and the transcript that closes it ends the wait the
// VAD started when it reported the speech over.
func (s *StreamService) PushFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	tf, isTranscript := f.(*frames.TranscriptionFrame)
	if isTranscript {
		if s.takeFinalizePending() {
			tf.Finalized = true
		}
		// Opened before the push so that the metrics frame the timing raises,
		// which is pushed from inside this call, finds its span already open.
		s.tracer.open()
		s.ttfb.transcript(ctx, tf.Finalized)
	}
	err := s.Base.PushFrame(ctx, f, dir)
	if isTranscript {
		s.tracer.record(tf)
		// Reported after the transcript rather than before it: the work the
		// measurement covers is not done until the transcript is out.
		s.work.report(ctx)
	}
	return err
}

// ServiceMetadataFrame implements service.MetadataDescriber, describing this
// transcriber to the rest of the pipeline, enriched from the Connector when it
// implements Describer.
func (s *StreamService) ServiceMetadataFrame() frames.ServiceMetadata {
	mf := frames.NewSTTMetadataFrame(0)
	mf.ServiceName = s.Name()
	var m Metadata
	if d, ok := s.conn.(Describer); ok {
		m = d.Metadata()
		mf.UserTurns = m.RecommendedUserTurns
	}
	mf.TTFSP99Latency = m.ttfs(s.Name())
	return mf
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
	var (
		stream Stream
		err    error
	)
	s.set.hold(func() { stream, err = s.conn.Connect(sessionCtx, s.sampleRate) })
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
	s.sendFailed = false
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

// sendAudio hands a chunk to the provider, unless the service can no longer do
// its job: one that cannot transcribe anything would drop the chunk anyway, and
// one that connects on demand would attempt a handshake per chunk to do it.
func (s *StreamService) sendAudio(audio []byte) {
	if !s.Usable() {
		return
	}
	s.send(audio)
}

// reportConnectionError puts a lost provider connection on the pipeline. It is
// not fatal: the call continues, and the application decides what losing
// transcription for part of it means.
func (s *StreamService) reportConnectionError(
	ctx context.Context, ef *frames.ErrorFrame, treatAsPermanent bool,
) {
	// Whatever was being measured ends here. The transcript it was waiting on is
	// not coming on this connection, and holding the measurement open would
	// carry it into the utterance after the one it belongs to.
	s.ttfb.reportNow(ctx)
	s.work.report(ctx)
	s.PushErrorFrame(ctx, ef, treatAsPermanent)
}

// recordUsage reports the audio the connection was given. The measurement covers
// the whole connection, which is what a stream-priced provider bills, so it is
// reported as usage rather than as a span: a span covering the connection would
// outlive every turn in it and could not sit under the one it belongs to.
func (s *StreamService) recordUsage(ctx context.Context, connectedAt time.Time, audioBytes int64) {
	_ = connectedAt
	audio := pcmDuration(audioBytes, s.sampleRate)
	if audio == 0 {
		return
	}
	s.mu.Lock()
	model := s.model
	s.mu.Unlock()
	metrics.RecordSTTAudio(ctx, s.Name(), model, audio.Seconds())
	s.pushUsageMetrics(ctx, audio)
}

// updateSettings merges an update into the provider's own settings and lets it
// act on what changed, reopening the session when the provider says the change
// cannot take effect on the one already running.
//
// The reopen is taken after the merge has released its lock, or it would
// deadlock against the dial it goes on to make.
func (s *StreamService) updateSettings(ctx context.Context, f *frames.STTUpdateSettingsFrame) {
	reopen, err := s.set.apply(ctx, f)
	if err != nil {
		s.PushError(ctx, "stt: settings update", err, false)
		return
	}
	if reopen {
		s.requestReopen(ctx)
	}
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
	failed := s.sendFailed
	// Stamped whether or not there is a session to send on: audio arriving is
	// what says the call is not idle, and a session that opens a moment later
	// has no catching up to do.
	s.lastAudio = time.Now()
	if stream != nil {
		s.audioBytes += int64(len(audio))
	}
	s.mu.Unlock()
	if stream == nil || failed {
		return
	}
	if err := stream.Send(audio); err != nil {
		// The session is gone as far as the audio is concerned. Reopening it is
		// the read loop's job, which the same failure ends; this only says so
		// once, and stops feeding a session that is not listening.
		slog.Warn("sending audio to the transcription session failed",
			"service", s.Name(), "err", err)
		s.mu.Lock()
		s.sendFailed = true
		s.mu.Unlock()
	}
}

// speechStarted tells the session a new utterance has begun, for a provider that
// carries text from one result to the next and has to be told the utterance it
// was building is over. It is a no-op for a provider that holds nothing between
// utterances, and for a session that is being replaced: the one that takes over
// starts empty anyway.
func (s *StreamService) speechStarted() {
	s.mu.Lock()
	stream := s.stream
	reopening := s.reopening
	s.mu.Unlock()
	if stream == nil || reopening {
		return
	}
	if ss, ok := stream.(SpeechStarter); ok {
		ss.SpeechStarted()
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
	// Recorded before the request goes out, so a provider that answers straight
	// away has something to be matched against. A provider that never confirms
	// leaves it standing, and it is cleared at the next utterance.
	s.mu.Lock()
	s.finalizeRequested = true
	s.mu.Unlock()
	if err := fin.Finalize(); err != nil {
		slog.Debug("stt finalize failed", "service", s.Name(), "err", err)
	}
}

// confirmFinalize takes the provider's word that it flushed what it was asked
// to, which marks the next transcript as the last one for the utterance. A
// confirmation for a finalize that was never asked for says nothing and is
// ignored.
func (s *StreamService) confirmFinalize() {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.finalizeRequested {
		s.finalizePending = true
		s.finalizeRequested = false
	}
}

// takeFinalizePending reports whether a confirmed finalize is waiting for the
// transcript it belongs to, and clears it so it marks only that one.
func (s *StreamService) takeFinalizePending() bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	pending := s.finalizePending
	s.finalizePending = false
	return pending
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
	s.emitSpeech(ctx, r)
	// Taken before the text is, since a provider with nothing left to say still
	// answers the finalize it was asked for, and that answer is what says the
	// transcript already sent closed the utterance.
	answered := r.Final && r.FromFinalize
	if answered {
		s.confirmFinalize()
	}
	if r.Text == "" {
		if answered {
			// The transcript went out ahead of the confirmation, so the wait
			// ended with it. Reported now rather than left to the deadline.
			s.ttfb.reportNow(ctx)
		}
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

// emitSpeech broadcasts the speech boundary a provider reported. The frames go
// out in pairs and only on a change: a provider repeating itself, or reporting a
// stop for speech that never started here, must not leave the pipeline holding a
// start that nothing closes.
//
// They are broadcast rather than pushed downstream because a turn beginning
// concerns both directions, and a barge-in on the provider's own detection has
// to reach the output to stop the bot as well as the aggregator to open the
// turn.
func (s *StreamService) emitSpeech(ctx context.Context, r Result) {
	switch r.Speech {
	case SpeechStarted:
		s.mu.Lock()
		already := s.speaking
		s.speaking = true
		s.mu.Unlock()
		if already {
			return
		}
		_ = s.Broadcast(ctx, func() frames.Frame { return frames.NewUserStartedSpeakingFrame() })
		if r.Interrupt {
			_ = s.Broadcast(ctx, func() frames.Frame { return frames.NewInterruptionFrame() })
		}
	case SpeechStopped:
		s.mu.Lock()
		speaking := s.speaking
		s.speaking = false
		s.mu.Unlock()
		if !speaking {
			return
		}
		_ = s.Broadcast(ctx, func() frames.Frame { return frames.NewUserStoppedSpeakingFrame() })
	case SpeechUnknown:
		// The provider leaves speech detection to the pipeline.
	}
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
