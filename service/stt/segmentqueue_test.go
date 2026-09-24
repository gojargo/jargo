package stt_test

import (
	"bytes"
	"context"
	"fmt"
	"sync"
	"testing"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/service/stt"
	"github.com/gojargo/jargo/utils/events"
)

// slowTranscriber takes delay over each segment and names the transcript after
// the segment's position, "segment 1" first.
type slowTranscriber struct {
	delay time.Duration

	mu  sync.Mutex
	got [][]byte
}

func (tr *slowTranscriber) Transcribe(ctx context.Context, audio []byte, _ int) (string, error) {
	tr.mu.Lock()
	tr.got = append(tr.got, append([]byte(nil), audio...))
	n := len(tr.got)
	tr.mu.Unlock()
	select {
	case <-time.After(tr.delay):
	case <-ctx.Done():
		return "", ctx.Err()
	}
	return fmt.Sprintf("segment %d", n), nil
}

// segmentFrames is one segment as the VAD delimits it.
func segmentFrames(pcm []byte) []frames.Frame {
	return []frames.Frame{
		frames.NewVADUserStartedSpeakingFrame(0, time.Time{}),
		frames.NewInputAudioRawFrame(pcm, 16000, 1),
		frames.NewVADUserStoppedSpeakingFrame(0, time.Time{}),
	}
}

// runSegments queues fs through svc, stops the pipeline gracefully, and returns
// the kinds of frame that reached the end of it, transcripts by their text.
func runSegments(t *testing.T, svc *stt.SegmentService, fs []frames.Frame) []string {
	t.Helper()
	var (
		mu   sync.Mutex
		seen []string
	)
	task := pipeline.NewWorker(pipeline.New(svc), pipeline.WorkerConfig{
		ReachedDownstreamFilter: pipeline.AnyFrame,
	})
	events.On(&task.Registry, pipeline.EventFrameReachedDownstream, func(_ context.Context, f frames.Frame) {
		mu.Lock()
		defer mu.Unlock()
		switch fr := f.(type) {
		case *frames.TranscriptionFrame:
			seen = append(seen, fr.Text)
		case *frames.InputAudioRawFrame:
			seen = append(seen, "audio")
		}
	})
	runDone := make(chan error, 1)
	go func() { runDone <- task.Run(context.Background()) }()
	for _, f := range fs {
		task.QueueFrame(f)
	}
	task.StopWhenDone()
	select {
	case <-runDone:
	case <-time.After(5 * time.Second):
		t.Fatal("the pipeline did not stop")
	}
	mu.Lock()
	defer mu.Unlock()
	return seen
}

// A segmented service never holds the audio path while a transcript is
// pending, as a streaming one does not: audio sent after the VAD stop reaches
// the end of the pipeline before the segment's transcript does.
func TestAudioKeepsFlowingWhileASegmentIsTranscribed(t *testing.T) {
	svc := stt.NewSegment("FakeSegmentSTT", &slowTranscriber{delay: 200 * time.Millisecond}, 16000)
	svc.SetTrailingSilence(0)

	pcm := make([]byte, 960)
	fs := segmentFrames(pcm)
	for range 3 {
		fs = append(fs, frames.NewInputAudioRawFrame(pcm, 16000, 1))
	}
	seen := runSegments(t, svc, fs)

	want := []string{"audio", "audio", "audio", "audio", "segment 1"}
	if fmt.Sprint(seen) != fmt.Sprint(want) {
		t.Fatalf("frames = %q, want %q", seen, want)
	}
}

// Segments are transcribed one at a time in the order they were cut, and a
// graceful stop transcribes what is still queued before the service stops.
func TestSegmentsAreTranscribedInOrderAndFlushedOnStop(t *testing.T) {
	svc := stt.NewSegment("FakeSegmentSTT", &slowTranscriber{delay: 50 * time.Millisecond}, 16000)
	svc.SetTrailingSilence(0)

	pcm := []byte{1, 2, 3, 4}
	var fs []frames.Frame
	for range 3 {
		fs = append(fs, segmentFrames(pcm)...)
	}
	var transcripts []string
	for _, s := range runSegments(t, svc, fs) {
		if s != "audio" {
			transcripts = append(transcripts, s)
		}
	}

	want := []string{"segment 1", "segment 2", "segment 3"}
	if fmt.Sprint(transcripts) != fmt.Sprint(want) {
		t.Fatalf("transcripts = %q, want %q", transcripts, want)
	}
}

// Audio from before the VAD reported the speech is kept, up to a second of it,
// because the VAD reports the start of speech a little after it began.
func TestASegmentStartsWithTheSecondBeforeTheVAD(t *testing.T) {
	tr := &slowTranscriber{}
	svc := stt.NewSegment("FakeSegmentSTT", tr, 16000)
	svc.SetTrailingSilence(0)

	// 1.5 s of audio before the VAD start: only the last second is kept.
	before := make([]byte, 48000)
	for i := range before {
		before[i] = byte(i / 16000)
	}
	speech := []byte{9, 9, 9, 9}
	runSegments(t, svc, []frames.Frame{
		frames.NewInputAudioRawFrame(before, 16000, 1),
		frames.NewVADUserStartedSpeakingFrame(0, time.Time{}),
		frames.NewInputAudioRawFrame(speech, 16000, 1),
		frames.NewVADUserStoppedSpeakingFrame(0, time.Time{}),
	})

	tr.mu.Lock()
	defer tr.mu.Unlock()
	if len(tr.got) != 1 {
		t.Fatalf("transcribed %d segments, want 1", len(tr.got))
	}
	want := append(append([]byte(nil), before[len(before)-32000:]...), speech...)
	if !bytes.Equal(tr.got[0], want) {
		t.Fatalf("segment is %d bytes, want the last second before the VAD plus the speech (%d)",
			len(tr.got[0]), len(want))
	}
}

// A segment is cut by the VAD alone: the turn frames that follow it play no
// part, so a turn strategy waiting on the transcript is not waiting on itself.
func TestASegmentIsTranscribedWithoutTheTurnEnding(t *testing.T) {
	tr := &fakeTranscriber{text: "heard", got: make(chan []byte, 1)}
	svc := stt.NewSegment("FakeSegmentSTT", tr, 16000)
	svc.SetTrailingSilence(0)

	task := pipeline.NewWorker(pipeline.New(svc), pipeline.WorkerConfig{})
	runDone := make(chan error, 1)
	go func() { runDone <- task.Run(context.Background()) }()
	defer func() {
		task.StopWhenDone()
		<-runDone
	}()
	for _, f := range segmentFrames([]byte{1, 2, 3, 4}) {
		task.QueueFrame(f)
	}

	select {
	case <-tr.got:
	case <-time.After(3 * time.Second):
		t.Fatal("the segment was not transcribed on the VAD stop")
	}
}
