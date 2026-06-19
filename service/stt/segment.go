package stt

import (
	"context"
	"sync"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor"
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
	*processor.Base
	tr      Transcriber
	cfgRate int

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
	s.Base = processor.New(name, s)
	return s
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
		return s.PushFrame(ctx, f, dir)
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
	default:
		return s.PushFrame(ctx, f, dir)
	}
}

// Cleanup waits for any in-flight transcription before tearing down.
func (s *SegmentService) Cleanup(ctx context.Context) error {
	s.wg.Wait()
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
		text, err := s.tr.Transcribe(ctx, audio, rate)
		if err != nil {
			if ctx.Err() == nil {
				s.PushError(ctx, "stt transcription failed", err, false)
			}
			return
		}
		if text == "" {
			return
		}
		tf := frames.NewTranscriptionFrame(text, "", time.Now().UTC().Format(time.RFC3339))
		tf.Finalized = true
		_ = s.PushFrame(ctx, tf, processor.Downstream)
	})
}
