package deepgram

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"
	"time"

	"github.com/coder/websocket"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/utils/events"
)

// dgResults is a live transcription result as Deepgram sends it.
func dgResults(transcript string, isFinal, fromFinalize bool) map[string]any {
	return map[string]any{
		"type": msgResults, "is_final": isFinal, "from_finalize": fromFinalize,
		"channel": map[string]any{"alternatives": []map[string]any{{"transcript": transcript}}},
	}
}

// finalizingServer is a fake Deepgram live endpoint that sends onAudio once the
// first audio arrives, and onFinalize when it is asked to finalize.
func finalizingServer(t *testing.T, onAudio, onFinalize []map[string]any) string {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		c, err := websocket.Accept(w, r, nil)
		if err != nil {
			return
		}
		defer func() { _ = c.CloseNow() }()
		send := func(msgs []map[string]any) {
			for _, m := range msgs {
				data, _ := json.Marshal(m)
				_ = c.Write(r.Context(), websocket.MessageText, data)
			}
		}
		heard := false
		for {
			typ, data, err := c.Read(r.Context())
			if err != nil {
				return
			}
			switch {
			case typ == websocket.MessageBinary && !heard:
				heard = true
				send(onAudio)
			case typ == websocket.MessageText && string(data) == `{"type":"Finalize"}`:
				send(onFinalize)
			}
		}
	}))
	t.Cleanup(srv.Close)
	return srv.URL
}

// transcriptionRun drives a Deepgram service against a fake endpoint and
// records the transcription frames that reach the end of the pipeline.
type transcriptionRun struct {
	task *pipeline.Worker
	done chan error

	mu   sync.Mutex
	seen []frames.Frame
	got  chan frames.Frame
}

func runTranscription(t *testing.T, endpoint string) *transcriptionRun {
	t.Helper()
	svc := NewSTT(sttConfigFor(endpoint))
	run := &transcriptionRun{done: make(chan error, 1), got: make(chan frames.Frame, 16)}
	run.task = pipeline.NewWorker(pipeline.New(svc), pipeline.WorkerConfig{ReachedDownstreamFilter: pipeline.AnyFrame})
	events.On(&run.task.Registry, pipeline.EventFrameReachedDownstream, func(_ context.Context, f frames.Frame) {
		switch f.(type) {
		case *frames.TranscriptionFrame, *frames.InterimTranscriptionFrame, *frames.TranscriptionFinalizedFrame:
			run.mu.Lock()
			run.seen = append(run.seen, f)
			run.mu.Unlock()
			run.got <- f
		}
	})
	go func() { run.done <- run.task.Run(context.Background()) }()
	// Audio sent before the session is open is held for it, and a finalize asked
	// of no session is not sent at all.
	deadline := time.Now().Add(3 * time.Second)
	for !svc.Connected() {
		if time.Now().After(deadline) {
			t.Fatal("the session never opened")
		}
		time.Sleep(5 * time.Millisecond)
	}
	t.Cleanup(func() {
		run.task.StopWhenDone()
		select {
		case <-run.done:
		case <-time.After(5 * time.Second):
			t.Error("pipeline did not shut down")
		}
	})
	return run
}

// waitFor returns the next transcription frame of the given kind, failing the
// test if none arrives.
func waitFor[F frames.Frame](t *testing.T, run *transcriptionRun) F {
	t.Helper()
	deadline := time.After(3 * time.Second)
	for {
		select {
		case f := <-run.got:
			if want, ok := f.(F); ok {
				return want
			}
		case <-deadline:
			var zero F
			t.Fatalf("no %T arrived", zero)
			return zero
		}
	}
}

// pushed lists what reached the end of the pipeline, as kind, text and whether
// it was finalized.
func (r *transcriptionRun) pushed() string {
	r.mu.Lock()
	defer r.mu.Unlock()
	var out []string
	for _, f := range r.seen {
		switch fr := f.(type) {
		case *frames.TranscriptionFrame:
			out = append(out, fmt.Sprintf("transcript(%s, finalized=%t)", fr.Text, fr.Finalized))
		case *frames.InterimTranscriptionFrame:
			out = append(out, fmt.Sprintf("interim(%s)", fr.Text))
		case *frames.TranscriptionFinalizedFrame:
			out = append(out, "finalized")
		}
	}
	return fmt.Sprint(out)
}

// Deepgram's endpointing can send the transcript before it is asked to
// finalize. The finalize response then has no text. The transcript already
// pushed cannot be marked finalized any more, so the confirmation is pushed as a
// frame of its own.
func TestEmptyFinalizeResponseFinalizesTheTranscriptBeforeIt(t *testing.T) {
	endpoint := finalizingServer(t,
		[]map[string]any{dgResults("Hello there.", true, false)},
		[]map[string]any{dgResults("", true, true)})
	run := runTranscription(t, endpoint)

	run.task.QueueFrame(frames.NewVADUserStartedSpeakingFrame(0.2, time.Now()))
	run.task.QueueFrame(frames.NewUserAudioRawFrame("session-user", make([]byte, 640), 16000, 1))
	waitFor[*frames.TranscriptionFrame](t, run)
	// The VAD stop asks Deepgram to finalize.
	run.task.QueueFrame(frames.NewVADUserStoppedSpeakingFrame(0.2, time.Now()))
	confirmation := waitFor[*frames.TranscriptionFinalizedFrame](t, run)

	if got, want := run.pushed(), "[transcript(Hello there., finalized=false) finalized]"; got != want {
		t.Fatalf("frames = %s, want %s", got, want)
	}
	if confirmation.UserID != "session-user" {
		t.Errorf("confirmation user = %q, want session-user", confirmation.UserID)
	}
}

func TestFinalizeResponseWithTextFinalizesItsOwnTranscript(t *testing.T) {
	endpoint := finalizingServer(t, nil, []map[string]any{
		dgResults("Hello there.", true, true),
		// Marks the end of what answers the finalize.
		dgResults("next", false, false),
	})
	run := runTranscription(t, endpoint)

	run.task.QueueFrame(frames.NewVADUserStartedSpeakingFrame(0.2, time.Now()))
	run.task.QueueFrame(frames.NewInputAudioRawFrame(make([]byte, 640), 16000, 1))
	run.task.QueueFrame(frames.NewVADUserStoppedSpeakingFrame(0.2, time.Now()))
	waitFor[*frames.InterimTranscriptionFrame](t, run)

	if got, want := run.pushed(), "[transcript(Hello there., finalized=true) interim(next)]"; got != want {
		t.Fatalf("frames = %s, want %s", got, want)
	}
}

// A confirmation that arrives after the user spoke again belongs to no
// utterance.
func TestEmptyFinalizeResponseNobodyAskedForFinalizesNothing(t *testing.T) {
	endpoint := finalizingServer(t, []map[string]any{
		dgResults("", true, true),
		// Marks the end of what the session sent.
		dgResults("next", false, false),
	}, nil)
	run := runTranscription(t, endpoint)

	run.task.QueueFrame(frames.NewInputAudioRawFrame(make([]byte, 640), 16000, 1))
	waitFor[*frames.InterimTranscriptionFrame](t, run)

	if got, want := run.pushed(), "[interim(next)]"; got != want {
		t.Fatalf("frames = %s, want %s", got, want)
	}
}
