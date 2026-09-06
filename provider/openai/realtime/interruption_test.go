package realtime_test

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/coder/websocket"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/provider/openai/realtime"
	"github.com/gojargo/jargo/utils/events"
)

// scriptedRealtime is a WebSocket server that replays the given server events
// and records every client message, so a test can assert on what the service
// said back.
type scriptedRealtime struct {
	*httptest.Server
	mu   sync.Mutex
	sent []map[string]any
}

func newScriptedRealtime(t *testing.T, serverEvents []string) *scriptedRealtime {
	t.Helper()
	f := &scriptedRealtime{}
	f.Server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		c, err := websocket.Accept(w, r, &websocket.AcceptOptions{InsecureSkipVerify: true})
		if err != nil {
			return
		}
		defer func() { _ = c.Close(websocket.StatusNormalClosure, "") }()
		ctx := r.Context()
		go func() {
			for {
				_, data, err := c.Read(ctx)
				if err != nil {
					return
				}
				var msg map[string]any
				if json.Unmarshal(data, &msg) != nil {
					continue
				}
				f.mu.Lock()
				f.sent = append(f.sent, msg)
				f.mu.Unlock()
			}
		}()
		for _, e := range serverEvents {
			if c.Write(ctx, websocket.MessageText, []byte(e)) != nil {
				return
			}
		}
		<-ctx.Done()
	}))
	t.Cleanup(f.Close)
	return f
}

func (f *scriptedRealtime) wsURL() string { return "ws" + strings.TrimPrefix(f.URL, "http") }

func (f *scriptedRealtime) messages() []map[string]any {
	f.mu.Lock()
	defer f.mu.Unlock()
	return append([]map[string]any(nil), f.sent...)
}

// awaitMessage waits for a client message of the given type and returns it.
func (f *scriptedRealtime) awaitMessage(t *testing.T, want string) map[string]any {
	t.Helper()
	deadline := time.Now().Add(3 * time.Second)
	for time.Now().Before(deadline) {
		for _, m := range f.messages() {
			if m["type"] == want {
				return m
			}
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatalf("no %q message was sent; got %v", want, f.messages())
	return nil
}

// runRealtime starts the service in a pipeline, collecting what reaches either
// end of it: an error and a transcript travel upstream, the rest downstream.
func runRealtime(t *testing.T, url string) (*pipeline.Worker, chan error, func() []frames.Frame) {
	t.Helper()
	return runRealtimeCollecting(t, url, true)
}

// runRealtimeCollecting starts the service, watching the far end of the pipeline
// and, when upstream is set, the near end too. A broadcast builds one frame per
// direction, so a test counting frames watches one end.
func runRealtimeCollecting(t *testing.T, url string, upstream bool) (
	*pipeline.Worker, chan error, func() []frames.Frame,
) {
	t.Helper()
	svc := realtime.New(realtime.Config{APIKey: "k", BaseURL: url})
	var mu sync.Mutex
	var got []frames.Frame
	cfg := pipeline.WorkerConfig{
		ReachedDownstreamFilter: pipeline.AnyFrame,
		Params:                  pipeline.Params{AudioInSampleRate: 24000, AudioOutSampleRate: 24000},
	}
	if upstream {
		cfg.ReachedUpstreamFilter = pipeline.AnyFrame
	}
	task := pipeline.NewWorker(pipeline.New(svc), cfg)
	record := func(_ context.Context, f frames.Frame) {
		mu.Lock()
		got = append(got, f)
		mu.Unlock()
	}
	events.On(&task.Registry, pipeline.EventFrameReachedDownstream, record)
	if upstream {
		events.On(&task.Registry, pipeline.EventFrameReachedUpstream, record)
	}
	done := make(chan error, 1)
	go func() { done <- task.Run(context.Background()) }()
	return task, done, func() []frames.Frame {
		mu.Lock()
		defer mu.Unlock()
		return append([]frames.Frame(nil), got...)
	}
}

// playingResponse scripts a session that has started speaking, so there is an
// assistant audio item an interruption can cut back.
func playingResponse(extra ...string) []string {
	audio := base64.StdEncoding.EncodeToString(make([]byte, 48000)) // 1s at 24 kHz
	return append([]string{
		`{"type":"response.created"}`,
		`{"type":"response.audio.delta","item_id":"item-audio","content_index":0,"delta":"` + audio + `"}`,
	}, extra...)
}

// TestSpeechStartedTruncatesTheAudioBeingPlayed checks the session's record of
// the turn is cut back to what the caller heard. Left whole, the model believes
// it said an answer the user cut off after a word, and answers what it never
// said next.
func TestSpeechStartedTruncatesTheAudioBeingPlayed(t *testing.T) {
	srv := newScriptedRealtime(t, playingResponse(`{"type":"input_audio_buffer.speech_started"}`))
	task, done, _ := runRealtime(t, srv.wsURL())

	truncate := srv.awaitMessage(t, "conversation.item.truncate")
	if truncate["item_id"] != "item-audio" {
		t.Errorf("item_id = %v, want the item being played", truncate["item_id"])
	}
	if truncate["content_index"] != float64(0) {
		t.Errorf("content_index = %v, want 0", truncate["content_index"])
	}
	// A second of audio arrived, so the cut is however much of it had played:
	// somewhere between nothing and the whole second.
	end, ok := truncate["audio_end_ms"].(float64)
	if !ok || end < 0 || end > 1000 {
		t.Errorf("audio_end_ms = %v, want the played part of one second", truncate["audio_end_ms"])
	}

	task.StopWhenDone()
	<-done
}

// TestInterruptionTruncatesTheAudioBeingPlayed checks the cut is also made where
// the pipeline reports the interruption, which is how a turn detected outside
// the session reaches it.
func TestInterruptionTruncatesTheAudioBeingPlayed(t *testing.T) {
	srv := newScriptedRealtime(t, playingResponse())
	task, done, collected := runRealtime(t, srv.wsURL())
	waitForFrame[*frames.TTSAudioRawFrame](t, collected)

	task.QueueFrame(frames.NewInterruptionFrame())
	truncate := srv.awaitMessage(t, "conversation.item.truncate")
	if truncate["item_id"] != "item-audio" {
		t.Errorf("item_id = %v, want the item being played", truncate["item_id"])
	}

	task.StopWhenDone()
	<-done
}

// TestBenignErrorsLeaveTheSessionRunning checks the races the interruption path
// runs into by design are not reported as failures. Canceling a response that
// has already finished says the model had stopped, which is what was asked for.
func TestBenignErrorsLeaveTheSessionRunning(t *testing.T) {
	srv := newScriptedRealtime(t, []string{
		`{"type":"error","error":{"code":"response_cancel_not_active","message":"no active response"}}`,
		`{"type":"response.created"}`,
	})
	task, done, collected := runRealtime(t, srv.wsURL())

	// The session carried on: an event after the error still reached the
	// pipeline, which it would not have had the read loop stopped.
	waitForFrame[*frames.BotStartedSpeakingFrame](t, collected)
	for _, f := range collected() {
		if _, ok := f.(*frames.ErrorFrame); ok {
			t.Error("a cancel race was reported as an error")
		}
	}

	task.StopWhenDone()
	<-done
}

// TestServerErrorIsReported checks a real failure reaches the pipeline.
func TestServerErrorIsReported(t *testing.T) {
	srv := newScriptedRealtime(t, []string{
		`{"type":"error","error":{"code":"invalid_api_key","message":"invalid api key"}}`,
	})
	task, done, collected := runRealtime(t, srv.wsURL())

	waitForFrame[*frames.ErrorFrame](t, collected)

	task.StopWhenDone()
	<-done
}

// TestFramesTravelOnce checks the service forwards a frame once. The LLM base
// forwards what it handles, so a service pushing again would deliver two of
// every frame, and describe itself to the pipeline twice with them.
func TestFramesTravelOnce(t *testing.T) {
	srv := newScriptedRealtime(t, nil)
	// Only what reaches the end of the pipeline: a broadcast builds a frame per
	// direction, so watching both ends would count the metadata twice.
	task, done, collected := runRealtimeCollecting(t, srv.wsURL(), false)
	srv.awaitMessage(t, "session.update")

	task.StopWhenDone()
	<-done

	starts, ends, metadata := 0, 0, 0
	for _, f := range collected() {
		switch f.(type) {
		case *frames.StartFrame:
			starts++
		case *frames.EndFrame:
			ends++
		case *frames.LLMServiceMetadataFrame:
			metadata++
		}
	}
	if starts != 1 {
		t.Errorf("StartFrames reaching downstream = %d, want 1", starts)
	}
	if ends != 1 {
		t.Errorf("EndFrames reaching downstream = %d, want 1", ends)
	}
	if metadata != 1 {
		t.Errorf("service metadata frames = %d, want 1", metadata)
	}
}

// TestInputAudioIsConsumed checks the audio the model is listening to does not
// travel on down the pipeline, where the output transport would play it back.
func TestInputAudioIsConsumed(t *testing.T) {
	srv := newScriptedRealtime(t, nil)
	task, done, collected := runRealtime(t, srv.wsURL())
	srv.awaitMessage(t, "session.update")

	task.QueueFrame(frames.NewInputAudioRawFrame([]byte{1, 2}, 24000, 1))
	srv.awaitMessage(t, "input_audio_buffer.append")

	task.StopWhenDone()
	<-done

	for _, f := range collected() {
		if _, ok := f.(*frames.InputAudioRawFrame); ok {
			t.Error("input audio traveled past the service that consumed it")
		}
	}
}

// waitForFrame blocks until a frame of type T has been collected.
func waitForFrame[T frames.Frame](t *testing.T, collected func() []frames.Frame) {
	t.Helper()
	deadline := time.Now().Add(3 * time.Second)
	for time.Now().Before(deadline) {
		for _, f := range collected() {
			if _, ok := f.(T); ok {
				return
			}
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatalf("no %T frame arrived", *new(T))
}

// TestUserTranscriptTravelsUpstream checks the user's words go towards the user
// aggregator, which sits before this service. Pushed downstream they would reach
// the assistant aggregator and the output transport instead, and the
// conversation would hold no record of what the user said.
func TestUserTranscriptTravelsUpstream(t *testing.T) {
	script := []string{
		`{"type":"conversation.item.input_audio_transcription.completed","transcript":"hi there"}`,
	}

	// Downstream only: nothing the user said should arrive here.
	srv := newScriptedRealtime(t, script)
	task, done, collected := runRealtimeCollecting(t, srv.wsURL(), false)
	srv.awaitMessage(t, "session.update")
	time.Sleep(200 * time.Millisecond)
	for _, f := range collected() {
		if _, ok := f.(*frames.TranscriptionFrame); ok {
			t.Error("the user's transcript traveled downstream, away from the user aggregator")
		}
	}
	task.StopWhenDone()
	<-done

	// Watching both ends, it does arrive.
	srv2 := newScriptedRealtime(t, script)
	task2, done2, collected2 := runRealtime(t, srv2.wsURL())
	waitForFrame[*frames.TranscriptionFrame](t, collected2)
	task2.StopWhenDone()
	<-done2
}

// TestInterruptionLeavesTheCancellingToTheSession checks what an interruption
// does not send. The session runs its own VAD, so it has already stopped the
// response it heard the user speak over: canceling again would race its own
// cancel, and clearing the input buffer would throw away the words the user is
// in the middle of saying.
func TestInterruptionLeavesTheCancellingToTheSession(t *testing.T) {
	srv := newScriptedRealtime(t, playingResponse())
	task, done, collected := runRealtime(t, srv.wsURL())
	waitForFrame[*frames.TTSAudioRawFrame](t, collected)

	task.QueueFrame(frames.NewInterruptionFrame())
	srv.awaitMessage(t, "conversation.item.truncate")

	for _, m := range srv.messages() {
		switch m["type"] {
		case "response.cancel":
			t.Error("the response was canceled by hand, which the session does itself")
		case "input_audio_buffer.clear":
			t.Error("the input buffer was cleared, losing the speech that interrupted")
		}
	}

	task.StopWhenDone()
	<-done
}
