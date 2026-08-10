package realtime_test

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/coder/websocket"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/provider/xai/realtime"
)

// fakeRealtime is a WebSocket server standing in for xAI's Realtime API. It
// records the handshake and every client message, and replays a scripted event
// sequence.
type fakeRealtime struct {
	*httptest.Server
	mu      sync.Mutex
	sent    []map[string]any
	query   url.Values
	headers http.Header
}

// newFakeRealtime starts a server that replays events after the client connects.
func newFakeRealtime(t *testing.T, events []string) *fakeRealtime {
	t.Helper()
	f := &fakeRealtime{}
	f.Server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		f.mu.Lock()
		f.query = r.URL.Query()
		f.headers = r.Header.Clone()
		f.mu.Unlock()

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

		for _, e := range events {
			if c.Write(ctx, websocket.MessageText, []byte(e)) != nil {
				return
			}
		}
		<-ctx.Done()
	}))
	t.Cleanup(f.Close)
	return f
}

// wsURL is the server's WebSocket address.
func (f *fakeRealtime) wsURL() string { return "ws" + strings.TrimPrefix(f.URL, "http") }

// messages returns a snapshot of what the client has sent so far.
func (f *fakeRealtime) messages() []map[string]any {
	f.mu.Lock()
	defer f.mu.Unlock()
	return append([]map[string]any(nil), f.sent...)
}

// awaitMessage waits for a client message of the given type and returns it.
func (f *fakeRealtime) awaitMessage(t *testing.T, want string) map[string]any {
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

// run starts the service in a pipeline, collecting the frames it pushes
// downstream.
func run(t *testing.T, cfg realtime.Config) (*pipeline.Task, chan error, func() []frames.Frame) {
	t.Helper()
	svc := realtime.New(cfg)

	var mu sync.Mutex
	var got []frames.Frame
	task := pipeline.NewTask(pipeline.New(svc), pipeline.TaskParams{
		AudioInSampleRate:       24000,
		AudioOutSampleRate:      24000,
		ReachedDownstreamFilter: pipeline.AnyFrame,
		OnReachedDownstream: func(f frames.Frame) {
			mu.Lock()
			got = append(got, f)
			mu.Unlock()
		},
	})
	done := make(chan error, 1)
	go func() { done <- task.Run(context.Background()) }()

	return task, done, func() []frames.Frame {
		mu.Lock()
		defer mu.Unlock()
		return append([]frames.Frame(nil), got...)
	}
}

// TestHandshake checks the model travels as a query parameter and the key as a
// Bearer token, which is how xAI selects the model for the session.
func TestHandshake(t *testing.T) {
	srv := newFakeRealtime(t, nil)
	task, done, _ := run(t, realtime.Config{
		APIKey:  "test-key",
		BaseURL: srv.wsURL(),
		Model:   "grok-voice-think-fast-1.0",
	})

	deadline := time.Now().Add(3 * time.Second)
	for time.Now().Before(deadline) {
		srv.mu.Lock()
		connected := srv.headers != nil
		srv.mu.Unlock()
		if connected {
			break
		}
		time.Sleep(10 * time.Millisecond)
	}

	srv.mu.Lock()
	query, headers := srv.query, srv.headers
	srv.mu.Unlock()

	if headers == nil {
		t.Fatal("the service never connected")
	}
	if got := query.Get("model"); got != "grok-voice-think-fast-1.0" {
		t.Errorf("model query = %q, want the configured model", got)
	}
	if got := headers.Get("Authorization"); got != "Bearer test-key" {
		t.Errorf("Authorization = %q, want the Bearer key", got)
	}

	task.StopWhenDone()
	<-done
}

// TestSessionConfiguredOnConversationCreated checks the service waits for the
// server to open the conversation before configuring the session. Configuring
// earlier is rejected by xAI.
func TestSessionConfiguredOnConversationCreated(t *testing.T) {
	srv := newFakeRealtime(t, nil)
	task, done, _ := run(t, realtime.Config{APIKey: "k", BaseURL: srv.wsURL()})

	// Give the service time to connect and, wrongly, configure unprompted.
	time.Sleep(300 * time.Millisecond)
	for _, m := range srv.messages() {
		if m["type"] == "session.update" {
			t.Fatal("session.update sent before the conversation was created")
		}
	}

	task.StopWhenDone()
	<-done
}

// TestStreamsEvents drives a full turn and checks each server event maps onto
// the frames the pipeline expects. The audio and transcript event names are
// xAI's, which differ from the older Realtime spelling.
func TestStreamsEvents(t *testing.T) {
	audio := []byte{1, 2, 3, 4, 5, 6}
	srv := newFakeRealtime(t, []string{
		`{"type":"conversation.created"}`,
		`{"type":"response.created"}`,
		`{"type":"response.output_audio.delta","delta":"` + base64.StdEncoding.EncodeToString(audio) + `"}`,
		`{"type":"response.output_audio_transcript.delta","delta":"hello"}`,
		`{"type":"input_audio_buffer.speech_started"}`,
		`{"type":"input_audio_buffer.speech_stopped"}`,
		`{"type":"conversation.item.input_audio_transcription.completed","transcript":"hi there"}`,
		`{"type":"response.done"}`,
	})

	task, done, collected := run(t, realtime.Config{APIKey: "k", BaseURL: srv.wsURL()})

	// Exercise the input path; the fake server records it.
	task.QueueFrame(frames.NewInputAudioRawFrame([]byte{7, 7}, 24000, 1))

	deadline := time.Now().Add(3 * time.Second)
	for time.Now().Before(deadline) && len(collected()) < 7 {
		time.Sleep(10 * time.Millisecond)
	}

	var (
		gotAudio                []byte
		botText, userTranscript string
		interrupted, userStart  bool
		userStop, botStart      bool
		botStop                 bool
		audioRate               int
	)
	for _, f := range collected() {
		switch fr := f.(type) {
		case *frames.TTSAudioRawFrame:
			gotAudio = fr.Audio
			audioRate = fr.SampleRate
		case *frames.LLMTextFrame:
			botText = fr.Text
		case *frames.TranscriptionFrame:
			userTranscript = fr.Text
		case *frames.InterruptionFrame:
			interrupted = true
		case *frames.UserStartedSpeakingFrame:
			userStart = true
		case *frames.UserStoppedSpeakingFrame:
			userStop = true
		case *frames.BotStartedSpeakingFrame:
			botStart = true
		case *frames.BotStoppedSpeakingFrame:
			botStop = true
		}
	}

	if string(gotAudio) != string(audio) {
		t.Errorf("bot audio = %v, want %v", gotAudio, audio)
	}
	if audioRate != 24000 {
		t.Errorf("bot audio sample rate = %d, want 24000", audioRate)
	}
	if botText != "hello" {
		t.Errorf("bot transcript = %q, want %q", botText, "hello")
	}
	if userTranscript != "hi there" {
		t.Errorf("user transcript = %q, want %q", userTranscript, "hi there")
	}
	if !interrupted || !userStart {
		t.Error("speech_started did not produce barge-in (interruption + user-started)")
	}
	if !userStop {
		t.Error("speech_stopped did not produce user-stopped-speaking")
	}
	if !botStart || !botStop {
		t.Error("response lifecycle did not produce bot started/stopped speaking")
	}

	// The session was configured once the conversation opened.
	session := srv.awaitMessage(t, "session.update")
	if _, ok := session["session"].(map[string]any); !ok {
		t.Errorf("session.update = %v, want a session object", session)
	}
	// Input audio reached the model.
	appended := srv.awaitMessage(t, "input_audio_buffer.append")
	if got, _ := appended["audio"].(string); got != base64.StdEncoding.EncodeToString([]byte{7, 7}) {
		t.Errorf("appended audio = %q, want the base64 PCM", got)
	}

	task.StopWhenDone()
	<-done
}

// TestSetToolsFrameReachesTheSession checks a tool change mid-conversation
// produces a session.update, since the running model never re-reads the context.
func TestSetToolsFrameReachesTheSession(t *testing.T) {
	srv := newFakeRealtime(t, []string{`{"type":"conversation.created"}`})
	task, done, _ := run(t, realtime.Config{APIKey: "k", BaseURL: srv.wsURL()})

	// Wait for the initial configuration so the later update is distinguishable.
	srv.awaitMessage(t, "session.update")

	task.QueueFrame(frames.NewLLMSetToolsFrame([]frames.Tool{{
		Name:        "get_weather",
		Description: "look up the weather",
		Parameters:  json.RawMessage(`{"type":"object"}`),
	}}))

	deadline := time.Now().Add(3 * time.Second)
	for time.Now().Before(deadline) {
		if sessionAdvertises(srv.messages(), "get_weather") {
			task.StopWhenDone()
			<-done
			return
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatalf("no session.update advertising get_weather was sent; got %v", srv.messages())
}

// TestContextFrameSeedsSessionTools checks the toolset on the shared context
// reaches the session without the caller pushing a tools frame by hand.
func TestContextFrameSeedsSessionTools(t *testing.T) {
	srv := newFakeRealtime(t, []string{`{"type":"conversation.created"}`})
	convo := frames.NewLLMContext("system")
	convo.SetTools([]frames.Tool{{Name: "book_table"}})

	task, done, _ := run(t, realtime.Config{APIKey: "k", BaseURL: srv.wsURL()})
	task.QueueFrame(frames.NewLLMContextFrame(convo))

	deadline := time.Now().Add(3 * time.Second)
	for time.Now().Before(deadline) {
		if sessionAdvertises(srv.messages(), "book_table") {
			task.StopWhenDone()
			<-done
			return
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatalf("no session.update advertising book_table was sent; got %v", srv.messages())
}

// sessionAdvertises reports whether any session.update carries a tool named want.
func sessionAdvertises(msgs []map[string]any, want string) bool {
	for _, m := range msgs {
		if m["type"] != "session.update" {
			continue
		}
		session, ok := m["session"].(map[string]any)
		if !ok {
			continue
		}
		tools, ok := session["tools"].([]any)
		if !ok {
			continue
		}
		for _, tool := range tools {
			if spec, ok := tool.(map[string]any); ok && spec["name"] == want {
				return true
			}
		}
	}
	return false
}
