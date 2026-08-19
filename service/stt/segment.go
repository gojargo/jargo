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

// SegmentService buffers a user's audio between UserStartedSpeakingFrame and
// UserStoppedSpeakingFrame, then transcribes the whole segment with a
// Transcriber. It requires a turn detector upstream (turntaking.Detector) to
// delimit segments; without those frames it never transcribes.
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
	mu         sync.Mutex
	buf        []byte
	speaking   bool
	wg         sync.WaitGroup
}

// NewSegment builds a segmented STT service named name driven by tr. A non-zero
// sampleRate overrides the transport's input rate.
func NewSegment(name string, tr Transcriber, sampleRate int) *SegmentService {
	s := &SegmentService{tr: tr, cfgRate: sampleRate}
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
			// A segmented service is handed the speech a turn detector cut out
			// for it, so voice activity detection is what drives it.
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

// ProcessFrame buffers speech audio and transcribes each completed segment.
func (s *SegmentService) ProcessFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	if err := s.Base.ProcessFrame(ctx, f, dir); err != nil {
		return err
	}
	switch fr := f.(type) {
	case *frames.StartFrame:
		s.sampleRate = s.cfgRate
		if s.sampleRate == 0 {
			s.sampleRate = fr.AudioInSampleRate
		}
		if err := s.PushFrame(ctx, f, dir); err != nil {
			return err
		}
		return nil
	case *frames.STTUpdateSettingsFrame:
		return s.handleSettings(ctx, fr, dir)
	case *frames.UserStartedSpeakingFrame:
		s.mu.Lock()
		s.buf = nil
		s.speaking = true
		s.mu.Unlock()
		return s.PushFrame(ctx, f, dir)
	case *frames.InputAudioRawFrame:
		s.mu.Lock()
		if s.speaking {
			s.buf = append(s.buf, fr.Audio...)
		}
		s.mu.Unlock()
		return s.PushFrame(ctx, f, dir)
	case *frames.UserStoppedSpeakingFrame:
		if err := s.PushFrame(ctx, f, dir); err != nil {
			return err
		}
		s.transcribe(ctx)
		return nil
	case *frames.VADUserStartedSpeakingFrame:
		s.ttfb.speechStarted()
		s.tracer.speechStarted(fr.SpeechStart())
		return s.PushFrame(ctx, f, dir)
	case *frames.VADUserStoppedSpeakingFrame:
		s.ttfb.speechEnded(ctx, fr)
		return s.PushFrame(ctx, f, dir)
	case *frames.InterruptionFrame:
		// The utterance being measured is not the one that matters any more.
		s.ttfb.interrupted()
		return s.PushFrame(ctx, f, dir)
	default:
		return s.PushFrame(ctx, f, dir)
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
		mf.UserTurns = m.RecommendedUserTurns
	}
	mf.TTFSP99Latency = m.ttfs(s.Name())
	return mf
}

// Cleanup waits for any in-flight transcription before tearing down.
func (s *SegmentService) Cleanup(ctx context.Context) error {
	s.wg.Wait()
	s.ttfb.close()
	s.tracer.close()
	return s.Base.Cleanup(ctx)
}

// transcribe hands the buffered segment to the Transcriber on its own goroutine
// so the input goroutine keeps flowing audio while the request is in flight.
func (s *SegmentService) transcribe(ctx context.Context) {
	s.mu.Lock()
	audio := s.buf
	rate := s.sampleRate
	s.buf = nil
	s.speaking = false
	s.mu.Unlock()
	if len(audio) == 0 {
		return
	}
	// A service that can no longer work cannot transcribe this segment. The
	// buffered audio is released above rather than growing for the rest of the
	// session.
	if !s.Usable() {
		return
	}
	s.wg.Go(func() {
		// The audio handed to the transcriber is what this segment is billed on,
		// and it is reported against the segment's own span, which the transcript
		// this call produces will open and close.
		played := pcmDuration(int64(len(audio)), rate)
		s.tracer.addUsage(frames.STTUsage{AudioSeconds: played.Seconds()})
		metrics.RecordSTTAudio(ctx, s.Name(), s.modelName(), played.Seconds())
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
	})
}

// CanGenerateMetrics reports that this service times transcription and reports
// the result, so the pipeline counts it when it collects the processors that
// report metrics.
func (s *SegmentService) CanGenerateMetrics() bool { return true }
