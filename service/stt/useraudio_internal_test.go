package stt

import (
	"bytes"
	"context"
	"log/slog"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
)

// answeringStream records the audio it is sent and answers with the results the
// test hands it, one batch at a time.
type answeringStream struct {
	kaStream
	out chan []Result
}

func (s *answeringStream) Recv() ([]Result, error) {
	select {
	case r := <-s.out:
		return r, nil
	case <-s.ctx.Done():
		return nil, s.ctx.Err()
	}
}

type answeringConnector struct{ stream *answeringStream }

func (c *answeringConnector) Connect(ctx context.Context, _ int) (Stream, error) {
	c.stream.ctx = ctx
	return c.stream, nil
}

// transcriptWatcher collects the transcripts reaching the end of the pipeline.
type transcriptWatcher struct {
	mu  sync.Mutex
	got []*frames.TranscriptionFrame
}

func (w *transcriptWatcher) observe(_ context.Context, f frames.Frame) {
	if tf, ok := f.(*frames.TranscriptionFrame); ok {
		w.mu.Lock()
		w.got = append(w.got, tf)
		w.mu.Unlock()
	}
}

func (w *transcriptWatcher) transcripts() []*frames.TranscriptionFrame {
	w.mu.Lock()
	defer w.mu.Unlock()
	return append([]*frames.TranscriptionFrame(nil), w.got...)
}

// Audio attributed to a user is input audio like any other: it is sent to the
// provider, and the transcript it produces names the user who spoke it. Audio
// naming nobody after it is attributed to nobody.
func TestStreamServiceTranscribesUserAudio(t *testing.T) {
	t.Parallel()

	st := &answeringStream{out: make(chan []Result, 1)}
	svc := NewStream("FakeSTT", &answeringConnector{stream: st}, 16000)
	w := &transcriptWatcher{}
	task, stop := runWatched(t, pipeline.New(svc), w.observe)
	defer stop()
	waitFor(t, "the session to open", func() bool { return svc.Connected() })

	task.QueueFrame(frames.NewUserAudioRawFrame("user-1", make([]byte, 640), 16000, 1))
	waitFor(t, "the user audio to be sent", func() bool { return len(st.sends()) == 1 })
	st.out <- []Result{{Text: "hello", Final: true, EndOfTurn: true}}
	waitFor(t, "the first transcript", func() bool { return len(w.transcripts()) == 1 })

	task.QueueFrame(frames.NewInputAudioRawFrame(make([]byte, 320), 16000, 1))
	waitFor(t, "the input audio to be sent", func() bool { return len(st.sends()) == 2 })
	st.out <- []Result{{Text: "again", Final: true, EndOfTurn: true}}
	waitFor(t, "the second transcript", func() bool { return len(w.transcripts()) == 2 })

	if got := chunkSizes(st.sends()); got[0] != 640 || got[1] != 320 {
		t.Errorf("sent chunks of %v bytes, want [640 320]", got)
	}
	got := w.transcripts()
	if got[0].UserID != "user-1" {
		t.Errorf("first transcript user = %q, want %q", got[0].UserID, "user-1")
	}
	if got[1].UserID != "" {
		t.Errorf("second transcript user = %q, want none", got[1].UserID)
	}
}

// Audio on its way out is the bot's, not the user's: it passes through the
// service untouched and is never sent to the provider.
func TestOutputAudioIsForwardedWithoutBeingTranscribed(t *testing.T) {
	t.Parallel()

	st := &kaStream{}
	svc := NewStream("FakeSTT", &kaConnector{stream: st}, 16000)
	var (
		mu   sync.Mutex
		seen []string
	)
	task, stop := runWatched(t, pipeline.New(svc), func(_ context.Context, f frames.Frame) {
		switch f.(type) {
		case *frames.InputAudioRawFrame:
			mu.Lock()
			seen = append(seen, "input")
			mu.Unlock()
		case *frames.OutputAudioRawFrame:
			mu.Lock()
			seen = append(seen, "output")
			mu.Unlock()
		}
	})
	defer stop()
	waitFor(t, "the session to open", func() bool { return svc.Connected() })

	input, output := bytes.Repeat([]byte{1, 0}, 160), bytes.Repeat([]byte{2, 0}, 160)
	task.QueueFrame(frames.NewInputAudioRawFrame(input, 16000, 1))
	task.QueueFrame(frames.NewOutputAudioRawFrame(output, 16000, 1))
	waitFor(t, "both frames downstream", func() bool {
		mu.Lock()
		defer mu.Unlock()
		return len(seen) == 2
	})

	mu.Lock()
	if seen[0] != "input" || seen[1] != "output" {
		t.Errorf("downstream = %v, want the input audio then the output audio", seen)
	}
	mu.Unlock()
	if sends := st.sends(); len(sends) != 1 || !bytes.Equal(sends[0], input) {
		t.Errorf("sent chunks of %v bytes, want only the input audio", chunkSizes(sends))
	}
}

// An audio frame carrying nothing is not sent to the provider, and is reported:
// a transport handing over empty frames is one whose audio is not getting
// through.
func TestAnEmptyAudioFrameIsNotSent(t *testing.T) {
	var logs bytes.Buffer
	prev := slog.Default()
	slog.SetDefault(slog.New(slog.NewTextHandler(&logs, &slog.HandlerOptions{Level: slog.LevelWarn})))
	t.Cleanup(func() { slog.SetDefault(prev) })

	st := &kaStream{}
	svc := NewStream("FakeSTT", &kaConnector{stream: st}, 16000)
	task, stop := runKeepaliveService(t, svc)
	defer stop()
	waitFor(t, "the session to open", func() bool { return svc.Connected() })

	task.QueueFrame(frames.NewInputAudioRawFrame(nil, 16000, 1))
	task.QueueFrame(frames.NewInputAudioRawFrame(make([]byte, 320), 16000, 1))
	waitFor(t, "the audio after the empty frame", func() bool { return len(st.sends()) > 0 })

	if sends := st.sends(); len(sends) != 1 || len(sends[0]) != 320 {
		t.Errorf("sent chunks of %v bytes, want only the 320 after the empty frame", chunkSizes(sends))
	}
	if !strings.Contains(logs.String(), "empty audio frame") {
		t.Errorf("logged %q, want the empty frame reported", logs.String())
	}
}

// userTranscriber records the segments it is given and answers with fixed text.
type userTranscriber struct{ got chan []byte }

func (tr *userTranscriber) Transcribe(_ context.Context, audio []byte, _ int) (string, error) {
	tr.got <- append([]byte(nil), audio...)
	return "hello", nil
}

// A segmented service buffers audio attributed to a user like any other input
// audio, and the transcript of the segment names the user who spoke it.
func TestSegmentServiceTranscribesUserAudio(t *testing.T) {
	t.Parallel()

	tr := &userTranscriber{got: make(chan []byte, 1)}
	svc := NewSegment("FakeSTT", tr, 16000)
	svc.SetTrailingSilence(0)
	w := &transcriptWatcher{}
	task, stop := runWatched(t, pipeline.New(svc), w.observe)
	defer stop()

	task.QueueFrame(frames.NewVADUserStartedSpeakingFrame(0, time.Time{}))
	task.QueueFrame(frames.NewUserAudioRawFrame("user-1", []byte{1, 2, 3, 4}, 16000, 1))
	task.QueueFrame(frames.NewVADUserStoppedSpeakingFrame(0, time.Time{}))

	select {
	case audio := <-tr.got:
		if len(audio) != 4 {
			t.Errorf("transcribed %d bytes, want the 4 the user spoke", len(audio))
		}
	case <-time.After(3 * time.Second):
		t.Fatal("the segment was never transcribed")
	}
	waitFor(t, "the transcript", func() bool { return len(w.transcripts()) == 1 })
	if got := w.transcripts()[0].UserID; got != "user-1" {
		t.Errorf("transcript user = %q, want %q", got, "user-1")
	}
}
