package deepgram

import (
	"context"
	"encoding/json"
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

// fluxSpeakRecorder is a fake Flux synthesis endpoint that records the text of
// every Speak it is sent and answers each turn with a little audio.
type fluxSpeakRecorder struct {
	mu    sync.Mutex
	spoke []string
}

func (r *fluxSpeakRecorder) serve(w http.ResponseWriter, req *http.Request) {
	c, err := websocket.Accept(w, req, nil)
	if err != nil {
		return
	}
	defer func() { _ = c.Close(websocket.StatusNormalClosure, "") }()
	ctx := req.Context()

	_, speak, err := c.Read(ctx)
	if err != nil {
		return
	}
	var msg fluxSpeak
	if json.Unmarshal(speak, &msg) == nil && msg.Type == "Speak" {
		r.mu.Lock()
		r.spoke = append(r.spoke, msg.Text)
		r.mu.Unlock()
	}
	// The Flush that ends the turn.
	if _, _, err := c.Read(ctx); err != nil {
		return
	}
	_ = c.Write(ctx, websocket.MessageBinary, []byte{1, 2, 3, 4})
	_ = c.Write(ctx, websocket.MessageText, mustJSON(map[string]any{
		"type": fluxTTSMsgSpeechMetadata, "speech_id": "s1", "audio_duration_ms": 10,
	}))
}

func (r *fluxSpeakRecorder) texts() []string {
	r.mu.Lock()
	defer r.mu.Unlock()
	return append([]string(nil), r.spoke...)
}

// Flux exists to start synthesizing on the first token rather than waiting for
// the sentence to finish, so the model's tokens reach it as written: one Speak
// each, with no space inserted between them and none taken away. Flux neither
// inserts nor strips whitespace between the units it is sent, so a space added
// here would split a word and a space removed would run two together.
func TestFluxTTSStreamsTokensAsWritten(t *testing.T) {
	rec := &fluxSpeakRecorder{}
	srv := httptest.NewServer(http.HandlerFunc(rec.serve))
	defer srv.Close()

	svc := NewFluxTTS(FluxTTSConfig{
		APIKey:     "k",
		SpeakURL:   wsURL(srv.URL),
		SampleRate: 24000,
	})
	task := pipeline.NewWorker(pipeline.New(svc), pipeline.WorkerConfig{
		ReachedDownstreamFilter: pipeline.AnyFrame,
	})
	stopped := make(chan struct{}, 1)
	events.On(&task.Registry, pipeline.EventFrameReachedDownstream, func(_ context.Context, f frames.Frame) {
		if _, ok := f.(*frames.TTSStoppedFrame); ok {
			select {
			case stopped <- struct{}{}:
			default:
			}
		}
	})

	runDone := make(chan error, 1)
	go func() { runDone <- task.Run(context.Background()) }()

	task.QueueFrame(frames.NewLLMFullResponseStartFrame())
	for _, token := range []string{"Unbelieva", "ble", " isn't it?"} {
		task.QueueFrame(frames.NewLLMTextFrame(token))
	}
	task.QueueFrame(frames.NewLLMFullResponseEndFrame())

	select {
	case <-stopped:
	case <-time.After(10 * time.Second):
		t.Fatal("the turn was never spoken")
	}
	task.StopWhenDone()
	select {
	case <-runDone:
	case <-time.After(10 * time.Second):
		t.Fatal("task did not finish")
	}

	got := rec.texts()
	want := []string{"Unbelieva", "ble", " isn't it?"}
	if len(got) != len(want) {
		t.Fatalf("spoke %q, want %q", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("spoke %q, want %q", got, want)
		}
	}
}
