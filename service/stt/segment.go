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
	"go.opentelemetry.io/otel/attribute"
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

	ttfb *ttfbTracker
	work *processingMeter

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
	s.ttfb = newTTFBTracker(s.Base.Base, func() string { return s.model })
	s.work = newProcessingMeter(s.Base.Base, func() string { return s.model })
	return s
}

// SetTTFBTimeout sets how long the service waits after the speech ends for the
// transcript that closes it, before reporting the latency against whatever
// arrived in the meantime. Zero restores DefaultTTFBTimeout.
func (s *SegmentService) SetTTFBTimeout(d time.Duration) { s.ttfb.setTimeout(d) }

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
func (s *SegmentService) PushFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	if tf, ok := f.(*frames.TranscriptionFrame); ok {
		s.ttfb.transcript(ctx, tf.Finalized)
	}
	return s.Base.PushFrame(ctx, f, dir)
}

// ServiceMetadataFrame implements service.MetadataDescriber, describing this
// transcriber to the rest of the pipeline, enriched from the Transcriber when it
// implements Describer.
func (s *SegmentService) ServiceMetadataFrame() frames.ServiceMetadata {
	mf := frames.NewSTTMetadataFrame(0)
	mf.ServiceName = s.Name()
	if d, ok := s.tr.(Describer); ok {
		m := d.Metadata()
		mf.UserTurns = m.RecommendedUserTurns
		mf.TTFSP99Latency = m.TTFSP99
	}
	return mf
}

// Cleanup waits for any in-flight transcription before tearing down.
func (s *SegmentService) Cleanup(ctx context.Context) error {
	s.wg.Wait()
	s.ttfb.close()
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
	s.wg.Go(func() {
		sctx, span := tracing.Tracer().Start(s.Tracing().Parent(ctx), "stt")
		defer span.End()
		played := pcmDuration(int64(len(audio)), rate)
		span.SetAttributes(
			attribute.String("stt.service", s.Name()),
			attribute.Int("stt.sample_rate", rate),
			attribute.Int64("stt.audio_ms", played.Milliseconds()),
		)
		tracing.SetSTTUsage(sctx, s.model, played)
		metrics.RecordSTTAudio(sctx, s.Name(), s.model, played.Seconds())
		s.pushUsageMetrics(sctx, played)

		start := time.Now()
		text, err := s.tr.Transcribe(sctx, audio, rate)
		if err != nil {
			if sctx.Err() == nil {
				span.RecordError(err)
				s.PushError(sctx, "stt transcription failed", err, false)
			}
			return
		}
		s.work.reportElapsed(sctx, time.Since(start))
		if text == "" {
			return
		}
		tf := frames.NewTranscriptionFrame(text, "", time.Now().UTC().Format(time.RFC3339))
		tf.Finalized = true
		_ = s.PushFrame(sctx, tf, processor.Downstream)
	})
}

// CanGenerateMetrics reports that this service times transcription and reports
// the result, so the pipeline counts it when it collects the processors that
// report metrics.
func (s *SegmentService) CanGenerateMetrics() bool { return true }
