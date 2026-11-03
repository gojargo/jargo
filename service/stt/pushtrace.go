package stt

import (
	"context"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/telemetry/tracing"
)

// TraceTranscripts traces the user's speech as a service that is not a
// transcription service reports it, a speech-to-speech service for one, with
// the same segment spans a transcription service records. The spans follow the
// frames the service pushes, so it needs no calls of its own:
//
//   - speech the VAD reports anchors the next span where the speech began;
//   - a transcript opens the span before it is pushed and is recorded on it
//     after, and a finalized one closes it;
//   - a time to first byte the service reports while a span is open is recorded
//     on it;
//   - the end of the user's turn closes a span no finalized transcript closed,
//     marking it incomplete.
//
// baseline supplies the attributes every span opens with, read as it opens.
func TraceTranscripts(p *processor.Base, baseline func() tracing.STTAttributes) {
	t := newSegmentTracer(p, baseline)
	p.Events().Add(processor.EventBeforePushFrame, func(_ context.Context, _ any, args ...any) {
		switch f := pushedFrame(args).(type) {
		case *frames.VADUserStartedSpeakingFrame:
			t.speechStarted(f.SpeechStart())
		case *frames.TranscriptionFrame:
			t.open()
		}
	})
	p.Events().Add(processor.EventAfterPushFrame, func(_ context.Context, _ any, args ...any) {
		switch f := pushedFrame(args).(type) {
		case *frames.UserStoppedSpeakingFrame:
			t.abandon(time.Time{})
		case *frames.MetricsFrame:
			for _, d := range f.Data {
				if ttfb, ok := d.(frames.TTFBMetricsData); ok {
					t.recordTTFB(ttfb.Value)
					break
				}
			}
		case *frames.TranscriptionFrame:
			t.record(f)
		}
	})
}

// pushedFrame is the frame a push event carries.
func pushedFrame(args []any) frames.Frame {
	if len(args) == 0 {
		return nil
	}
	f, _ := args[0].(frames.Frame)
	return f
}
