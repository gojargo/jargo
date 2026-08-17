package stt_test

import (
	"bytes"
	"context"
	"sync/atomic"
	"testing"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/service/stt"
	"github.com/gojargo/jargo/utils/events"
)

// countingStream counts the audio chunks it is sent and never produces a result.
type countingStream struct {
	sent atomic.Int64
	ctx  context.Context
}

func (s *countingStream) Send([]byte) error { s.sent.Add(1); return nil }

func (s *countingStream) Recv() ([]stt.Result, error) {
	<-s.ctx.Done()
	return nil, s.ctx.Err()
}

func (s *countingStream) Close() error { return nil }

type countingConnector struct{ stream *countingStream }

func (c *countingConnector) Connect(ctx context.Context, _ int) (stt.Stream, error) {
	c.stream.ctx = ctx
	return c.stream, nil
}

func (c *countingConnector) Metadata() stt.Metadata { return stt.Metadata{Model: "counting"} }

// streamAudio feeds two chunks of audio through a streaming service and returns
// how many reached the provider.
func streamAudio(t *testing.T, usable bool) int64 {
	t.Helper()
	conn := &countingConnector{stream: &countingStream{}}
	svc := stt.NewStream("CountingSTT", conn, 16000)
	if !usable {
		svc.SetUsable(context.Background(), false)
	}

	// The audio frames travel on whether or not they were sent to the provider,
	// so seeing the last of them arrive downstream is what says the service is
	// done with them.
	arrived := make(chan struct{}, 4)
	task := pipeline.NewWorker(pipeline.New(svc), pipeline.WorkerConfig{
		ReachedDownstreamFilter: pipeline.AnyFrame,
	})
	events.On(&task.Registry, pipeline.EventFrameReachedDownstream, func(_ context.Context, f frames.Frame) {
		if _, ok := f.(*frames.InputAudioRawFrame); ok {
			select {
			case arrived <- struct{}{}:
			default:
			}
		}
	})
	runDone := make(chan error, 1)
	go func() { runDone <- task.Run(context.Background()) }()

	for range 2 {
		task.QueueFrame(frames.NewInputAudioRawFrame(make([]byte, 320), 16000, 1))
	}
	for range 2 {
		select {
		case <-arrived:
		case <-time.After(3 * time.Second):
			t.Fatal("the audio never reached the end of the pipeline")
		}
	}
	task.StopWhenDone()
	<-runDone

	return conn.stream.sent.Load()
}

func TestAudioIsTranscribedWhileTheServiceIsHealthy(t *testing.T) {
	if got := streamAudio(t, true); got != 2 {
		t.Errorf("sent %d chunks, want 2", got)
	}
}

func TestAudioIsDroppedOnceTheServiceIsUnusable(t *testing.T) {
	if got := streamAudio(t, false); got != 0 {
		t.Errorf("sent %d chunks, want 0", got)
	}
}

func TestSegmentIsDroppedOnceTheServiceIsUnusable(t *testing.T) {
	tr := &fakeTranscriber{text: "never asked", got: make(chan []byte, 1)}
	svc := stt.NewSegment("CountingSegmentSTT", tr, 16000)
	svc.SetUsable(context.Background(), false)

	stopped := make(chan struct{}, 1)
	task := pipeline.NewWorker(pipeline.New(svc), pipeline.WorkerConfig{
		ReachedDownstreamFilter: pipeline.AnyFrame,
	})
	events.On(&task.Registry, pipeline.EventFrameReachedDownstream, func(_ context.Context, f frames.Frame) {
		if _, ok := f.(*frames.UserStoppedSpeakingFrame); ok {
			select {
			case stopped <- struct{}{}:
			default:
			}
		}
	})
	runDone := make(chan error, 1)
	go func() { runDone <- task.Run(context.Background()) }()

	task.QueueFrame(frames.NewUserStartedSpeakingFrame())
	task.QueueFrame(frames.NewInputAudioRawFrame([]byte{1, 2, 3, 4}, 16000, 1))
	task.QueueFrame(frames.NewUserStoppedSpeakingFrame())

	select {
	case <-stopped:
	case <-time.After(3 * time.Second):
		t.Fatal("the turn never finished")
	}

	select {
	case audio := <-tr.got:
		t.Fatalf("the transcriber was asked for % x", audio)
	case <-time.After(200 * time.Millisecond):
	}

	task.StopWhenDone()
	<-runDone
}

func TestABufferedSegmentIsReleasedRatherThanKept(t *testing.T) {
	// A service that cannot transcribe must not go on accumulating the audio it
	// will never be asked about.
	tr := &fakeTranscriber{text: "spoken", got: make(chan []byte, 2)}
	svc := stt.NewSegment("CountingSegmentSTT", tr, 16000)
	ctx := context.Background()
	svc.SetUsable(ctx, false)

	transcribed := make(chan struct{}, 1)
	task := pipeline.NewWorker(pipeline.New(svc), pipeline.WorkerConfig{
		ReachedDownstreamFilter: pipeline.AnyFrame,
	})
	events.On(&task.Registry, pipeline.EventFrameReachedDownstream, func(_ context.Context, f frames.Frame) {
		if _, ok := f.(*frames.TranscriptionFrame); ok {
			select {
			case transcribed <- struct{}{}:
			default:
			}
		}
	})
	runDone := make(chan error, 1)
	go func() { runDone <- task.Run(context.Background()) }()

	dropped := []byte{9, 9, 9, 9}
	task.QueueFrame(frames.NewUserStartedSpeakingFrame())
	task.QueueFrame(frames.NewInputAudioRawFrame(dropped, 16000, 1))
	task.QueueFrame(frames.NewUserStoppedSpeakingFrame())
	time.Sleep(200 * time.Millisecond)

	// Brought back, the next turn carries its own audio alone.
	svc.SetUsable(ctx, true)
	kept := []byte{1, 2, 3, 4}
	task.QueueFrame(frames.NewUserStartedSpeakingFrame())
	task.QueueFrame(frames.NewInputAudioRawFrame(kept, 16000, 1))
	task.QueueFrame(frames.NewUserStoppedSpeakingFrame())

	select {
	case got := <-tr.got:
		if !bytes.Equal(got, kept) {
			t.Fatalf("transcriber audio = % x, want % x", got, kept)
		}
	case <-time.After(3 * time.Second):
		t.Fatal("the service never transcribed the turn it could work on")
	}

	select {
	case <-transcribed:
	case <-time.After(3 * time.Second):
		t.Fatal("no transcript reached the pipeline")
	}
	task.StopWhenDone()
	<-runDone
}
