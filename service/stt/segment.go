package stt

import (
	"context"
	"sync"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/service"
	"github.com/gojargo/jargo/telemetry/metrics"
	"github.com/gojargo/jargo/telemetry/tracing"
)

// Transcriber turns a complete audio segment into text. The audio is 16-bit
// mono PCM at sampleRate.
type Transcriber interface {
	Transcribe(ctx context.Context, audio []byte, sampleRate int) (string, error)
}

// DefaultTrailingSilence is how much silence is appended to each segment before
// it is transcribed. A segment ends right where the detector stopped, and models
// tend to drop or garble the final word when the audio ends that abruptly.
const DefaultTrailingSilence = 500 * time.Millisecond

// SegmentService transcribes speech a segment at a time, using the VAD to find
// the segments: it buffers the audio between VADUserStartedSpeakingFrame and
// VADUserStoppedSpeakingFrame and hands each segment to a Transcriber whole. It
// requires voice activity detection in the pipeline; without the VAD frames it
// never transcribes.
//
// The VAD reports the start of speech a little after it began, so while the
// user is not speaking the service keeps the last second of audio, and a
// segment starts with it.
//
// A segment ends right where the VAD stopped, so each one is padded with
// trailing silence before transcription and the model hears the end of speech.
//
// Transcription runs off the audio path. When the VAD stops, the segment is
// queued and one background task transcribes the queued segments in order,
// pushing each transcript as it completes, while audio and other frames keep
// flowing through the service. An EndFrame transcribes what is queued before it
// goes on; a CancelFrame drops it.
type SegmentService struct {
	*service.Base
	tr      Transcriber
	cfgRate int
	model   string

	ttfb   *ttfbTracker
	work   *processingMeter
	tracer *segmentTracer

	// set applies settings updates to the provider's own store.
	set *providerSettings

	sampleRate int
	// trailingSilence is how much silence is appended to a segment before it is
	// transcribed. It is settable so a caller can turn the padding off.
	trailingSilence time.Duration

	mu sync.Mutex
	// buf is the audio of the segment being gathered. While the user is not
	// speaking it holds at most a second, preRoll bytes.
	buf      []byte
	preRoll  int
	speaking bool

	// queue holds the segments waiting to be transcribed, in the order they
	// were cut. queued is signaled when one is added.
	queue  [][]byte
	queued chan struct{}
	// finishing tells the segment task to stop once the queue is empty.
	finishing bool
	// segmentCancel stops the segment task at once, nil when none is running.
	segmentCancel context.CancelFunc
	segmentWG     sync.WaitGroup
}

// CurrentSettings is the service's current settings, for code outside the
// service to read, or nil when its provider keeps none. They are still changed
// through an STTUpdateSettingsFrame.
func (s *SegmentService) CurrentSettings() any { return s.set.settings() }

// NewSegment builds a segmented STT service named name driven by tr. A non-zero
// sampleRate overrides the transport's input rate.
func NewSegment(name string, tr Transcriber, sampleRate int) *SegmentService {
	s := &SegmentService{
		tr: tr, cfgRate: sampleRate, trailingSilence: DefaultTrailingSilence,
		queued: make(chan struct{}, 1),
	}
	if d, ok := tr.(Describer); ok {
		s.model = d.Metadata().Model
	}
	s.Base = service.New(name, s)
	s.set = &providerSettings{provider: tr, name: s.Name, onModel: s.setModel, onChanged: s.SettingsUpdated}
	s.ttfb = newTTFBTracker(s.Base.Base, s.modelName)
	s.work = newProcessingMeter(s.Base.Base, s.modelName)
	s.tracer = newSegmentTracer(s.Base.Base, func() tracing.STTAttributes {
		return tracing.STTAttributes{
			Service:  s.TypeName(),
			Model:    s.modelName(),
			Settings: s.set.traceSettings(),
			// A segmented service transcribes the speech the VAD cut out, so
			// voice activity detection is what drives it.
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

// SetTrailingSilence sets how much silence is appended to each segment before it
// is transcribed, so the model hears the end of speech and can finish the last
// word. Set it to zero to send the segment exactly as it was cut.
//
// The padding is part of what the provider receives, so it is part of what usage
// is measured on.
func (s *SegmentService) SetTrailingSilence(d time.Duration) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if d < 0 {
		d = 0
	}
	s.trailingSilence = d
}

// SetTTFBTimeout sets how long the service waits after the speech ends for the
// transcript that closes it, before reporting the latency against whatever
// arrived in the meantime. Zero restores DefaultTTFBTimeout.
func (s *SegmentService) SetTTFBTimeout(d time.Duration) { s.ttfb.setTimeout(d) }

// STTService marks this processor as a speech-to-text service. See
// StreamService.STTService.
func (s *SegmentService) STTService() {}

// modelName is the model in force, which labels what this service reports.
func (s *SegmentService) modelName() string {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.model
}

// setModel relabels what this service reports with the model now in force.
func (s *SegmentService) setModel(model string) {
	s.mu.Lock()
	s.model = model
	s.mu.Unlock()
}

// handleSettings applies an update meant for this service. One naming another
// service is left untouched and travels on, so the service it was meant for
// gets it.
func (s *SegmentService) handleSettings(
	ctx context.Context, f *frames.STTUpdateSettingsFrame, dir processor.Direction,
) error {
	if !f.TargetsService(s) {
		return s.PushFrame(ctx, f, dir)
	}
	s.updateSettings(ctx, f)
	return nil
}

// updateSettings merges an update into the provider's own settings and lets it
// act on what changed.
//
// A provider may ask for its session to be replaced, which a streaming service
// does by reopening the connection. There is no session here: each segment is
// transcribed on its own, so the next one simply reads the settings as they now
// stand, and the request is nothing this service has to act on.
func (s *SegmentService) updateSettings(ctx context.Context, f *frames.STTUpdateSettingsFrame) {
	if _, err := s.set.apply(ctx, f); err != nil {
		s.PushError(ctx, "stt: settings update", err, false)
	}
}

// Setup resolves the rate the service transcribes at. A rate configured on the
// service wins; otherwise it takes the pipeline's input rate, which it knows
// from the moment it is set up rather than when the StartFrame arrives.
func (s *SegmentService) Setup(ctx context.Context, st processor.Setup) error {
	if err := s.Base.Setup(ctx, st); err != nil {
		return err
	}
	s.sampleRate = s.cfgRate
	if s.sampleRate == 0 {
		s.sampleRate = st.AudioInSampleRate
	}
	// One second of 16-bit mono audio.
	s.preRoll = s.sampleRate * 2
	return nil
}

// ProcessFrame buffers speech audio and queues each completed segment for
// transcription.
func (s *SegmentService) ProcessFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	if err := s.Base.ProcessFrame(ctx, f, dir); err != nil {
		return err
	}
	switch fr := f.(type) {
	case *frames.StartFrame:
		s.startSegmentTask(ctx)
		return s.PushFrame(ctx, f, dir)
	case *frames.EndFrame:
		// What was queued is transcribed before the service stops.
		s.stopSegmentTask(true)
		return s.PushFrame(ctx, f, dir)
	case *frames.CancelFrame:
		s.stopSegmentTask(false)
		return s.PushFrame(ctx, f, dir)
	case *frames.STTUpdateSettingsFrame:
		return s.handleSettings(ctx, fr, dir)
	case *frames.InputAudioRawFrame:
		s.bufferAudio(fr.Audio)
		return s.PushFrame(ctx, f, dir)
	case *frames.VADUserStartedSpeakingFrame:
		s.ttfb.speechStarted()
		s.tracer.speechStarted(fr.SpeechStart())
		s.mu.Lock()
		s.speaking = true
		s.mu.Unlock()
		return s.PushFrame(ctx, f, dir)
	case *frames.VADUserStoppedSpeakingFrame:
		s.ttfb.speechEnded(ctx, fr)
		if err := s.PushFrame(ctx, f, dir); err != nil {
			return err
		}
		s.endSegment()
		return nil
	case *frames.InterruptionFrame:
		// The utterance being measured is not the one that matters any more.
		s.ttfb.interrupted()
		return s.PushFrame(ctx, f, dir)
	default:
		return s.PushFrame(ctx, f, dir)
	}
}

// bufferAudio adds audio to the segment being gathered. While the user is
// speaking the buffer keeps growing; while they are not, only the last second
// is kept, to cover the delay between the speech starting and the VAD
// reporting it.
func (s *SegmentService) bufferAudio(audio []byte) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.buf = append(s.buf, audio...)
	if !s.speaking && len(s.buf) > s.preRoll {
		s.buf = append([]byte(nil), s.buf[len(s.buf)-s.preRoll:]...)
	}
}

// endSegment closes the segment the VAD just ended and queues it, padded with
// trailing silence, for the segment task to transcribe.
func (s *SegmentService) endSegment() {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.speaking = false
	audio := s.buf
	s.buf = nil
	// A service that can no longer work cannot transcribe this segment. The
	// buffered audio is released above rather than growing for the rest of the
	// session.
	if !s.Usable() {
		return
	}
	audio = append(audio, silence(s.trailingSilence, s.sampleRate)...)
	if len(audio) == 0 {
		return
	}
	s.queue = append(s.queue, audio)
	select {
	case s.queued <- struct{}{}:
	default:
	}
}

// startSegmentTask starts the task that transcribes queued segments, replacing
// any still running.
func (s *SegmentService) startSegmentTask(ctx context.Context) {
	s.stopSegmentTask(false)
	taskCtx, cancel := context.WithCancel(ctx)
	s.mu.Lock()
	s.segmentCancel = cancel
	s.finishing = false
	s.mu.Unlock()
	s.segmentWG.Go(func() { s.segmentTask(taskCtx) })
}

// stopSegmentTask stops the segment task and waits for it. Draining, the task
// first transcribes every segment already queued; otherwise they are dropped.
func (s *SegmentService) stopSegmentTask(drain bool) {
	s.mu.Lock()
	cancel := s.segmentCancel
	s.segmentCancel = nil
	s.finishing = true
	if !drain {
		s.queue = nil
	}
	s.mu.Unlock()
	if cancel == nil {
		return
	}
	select {
	case s.queued <- struct{}{}:
	default:
	}
	if !drain {
		cancel()
	}
	s.segmentWG.Wait()
	cancel()
}

// segmentTask transcribes queued segments one at a time, in the order they were
// cut, until it is told to finish and the queue is empty.
func (s *SegmentService) segmentTask(ctx context.Context) {
	for {
		s.mu.Lock()
		if len(s.queue) == 0 {
			finishing := s.finishing
			s.mu.Unlock()
			if finishing {
				return
			}
			select {
			case <-s.queued:
				continue
			case <-ctx.Done():
				return
			}
		}
		audio := s.queue[0]
		s.queue = s.queue[1:]
		rate := s.sampleRate
		s.mu.Unlock()
		s.transcribe(ctx, audio, rate)
		if ctx.Err() != nil {
			return
		}
	}
}

// PushFrame pushes a frame on, timing the transcripts on their way out: a
// segment's transcript closes the utterance it was cut from, and ends the wait
// the VAD started when it reported the speech over.
//
// The segment's span is opened before the transcript is pushed and written after
// it. Opening first is what lets the metrics frame that the timing raises, which
// is pushed from inside this call, find the span it belongs to already open.
func (s *SegmentService) PushFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	tf, isTranscript := f.(*frames.TranscriptionFrame)
	if isTranscript {
		s.tracer.open()
		s.ttfb.transcript(ctx, tf.Finalized)
	}
	err := s.Base.PushFrame(ctx, f, dir)
	if isTranscript {
		s.tracer.record(tf)
	}
	return err
}

// ServiceMetadataFrame implements service.MetadataDescriber, describing this
// transcriber to the rest of the pipeline, enriched from the Transcriber when it
// implements Describer.
func (s *SegmentService) ServiceMetadataFrame() frames.ServiceMetadata {
	mf := frames.NewSTTMetadataFrame(0)
	mf.ServiceName = s.Name()
	var m Metadata
	if d, ok := s.tr.(Describer); ok {
		m = d.Metadata()
		mf.UserTurnStrategies = m.UserTurnStrategies
	}
	mf.TTFSP99Latency = m.ttfs(s.Name())
	return mf
}

// Cleanup stops the segment task before tearing down.
func (s *SegmentService) Cleanup(ctx context.Context) error {
	s.stopSegmentTask(false)
	s.ttfb.close()
	s.tracer.close()
	return s.Base.Cleanup(ctx)
}

// transcribe hands one segment to the Transcriber and pushes its transcript.
func (s *SegmentService) transcribe(ctx context.Context, audio []byte, rate int) {
	// The audio handed to the transcriber is what this segment is billed on, and
	// it is reported before the transcription, against the segment's own span,
	// which the transcript this call produces will open and close.
	played := pcmDuration(int64(len(audio)), rate)
	s.tracer.addUsage(frames.STTUsage{AudioSeconds: played.Seconds()})
	metrics.RecordSTTAudio(ctx, s.TypeName(), s.modelName(), played.Seconds())
	s.pushUsageMetrics(ctx, played)

	start := time.Now()
	text, err := s.tr.Transcribe(ctx, audio, rate)
	if err != nil {
		if ctx.Err() == nil {
			s.tracer.recordError(err)
			s.PushError(ctx, "stt transcription failed", err, false)
		}
		return
	}
	s.work.reportElapsed(ctx, time.Since(start))
	if text == "" {
		return
	}
	tf := frames.NewTranscriptionFrame(text, "", frames.NowTimestamp())
	tf.Finalized = true
	_ = s.PushFrame(ctx, tf, processor.Downstream)
}

// silence is d of 16-bit mono silence at rate, as the samples a segment is
// padded with. A rate that is not known yet pads nothing: the length would be
// meaningless, and the transcriber is about to be handed audio it can measure
// for itself.
func silence(d time.Duration, rate int) []byte {
	if d <= 0 || rate <= 0 {
		return nil
	}
	samples := int(float64(rate) * d.Seconds())
	if samples <= 0 {
		return nil
	}
	return make([]byte, samples*2)
}

// CanGenerateMetrics reports that this service times transcription and reports
// the result, so the pipeline counts it when it collects the processors that
// report metrics.
func (s *SegmentService) CanGenerateMetrics() bool { return true }
