package stt

import (
	"context"
	"errors"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/utils/events"
)

// errSendRefused is a session refusing audio.
//
//nolint:gochecknoglobals // sentinel error
var errSendRefused = errors.New("socket closed")

// refusingStream refuses every send.
type refusingStream struct {
	idleStream
}

func (s *refusingStream) Send([]byte) error { return errSendRefused }

// reportingStream is a refusingStream that reports its failed sends.
type reportingStream struct{ refusingStream }

func (s *reportingStream) SendErrorMessage(err error) string { return "send error: " + err.Error() }

type oneStreamConnector struct{ st Stream }

func (c *oneStreamConnector) Connect(ctx context.Context, _ int) (Stream, error) {
	switch st := c.st.(type) {
	case *refusingStream:
		st.ctx = ctx
	case *reportingStream:
		st.ctx = ctx
	}
	return c.st, nil
}

// sendErrors runs audio through a streaming service over st and returns the
// errors it pushed.
func sendErrors(t *testing.T, st Stream) []string {
	t.Helper()
	s := NewStream("FakeSTT", &oneStreamConnector{st: st}, 16000)
	var mu sync.Mutex
	var errs []string
	task := pipeline.NewWorker(pipeline.New(s), pipeline.WorkerConfig{ReachedUpstreamFilter: pipeline.AnyFrame})
	events.On(&task.Registry, pipeline.EventFrameReachedUpstream, func(_ context.Context, f frames.Frame) {
		if ef, ok := f.(*frames.ErrorFrame); ok {
			mu.Lock()
			errs = append(errs, ef.Error)
			mu.Unlock()
		}
	})
	done := make(chan error, 1)
	go func() { done <- task.Run(context.Background()) }()
	time.Sleep(100 * time.Millisecond)
	for range 3 {
		task.QueueFrame(frames.NewInputAudioRawFrame(make([]byte, 320), 16000, 1))
	}
	time.Sleep(200 * time.Millisecond)
	task.StopWhenDone()
	<-done
	mu.Lock()
	defer mu.Unlock()
	return errs
}

// TestAFailedSendIsReportedWhenTheProviderAsks checks a provider that reports
// its failed sends has the failure pushed as an error, and one that does not
// only has it logged.
func TestAFailedSendIsReportedWhenTheProviderAsks(t *testing.T) {
	reported := sendErrors(t, &reportingStream{})
	if len(reported) != 1 || !strings.Contains(reported[0], "send error: socket closed") {
		t.Errorf("errors = %v, want the failed send reported once", reported)
	}
	if logged := sendErrors(t, &refusingStream{}); len(logged) != 0 {
		t.Errorf("errors = %v, want none from a provider that does not report", logged)
	}
}
