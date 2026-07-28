package realtime_test

import (
	"context"
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
)

// A realtime model generates continuously, so it never re-reads the shared LLM
// context between turns: a tool change reaches it only if the service pushes a
// session.update. These tests drive the service against a fake server and assert
// on what it actually sent.

// recordingRealtime is a WebSocket server that records every message the client
// sends and never speaks first.
type recordingRealtime struct {
	*httptest.Server
	mu   sync.Mutex
	sent []map[string]any
}

func newRecordingRealtime(t *testing.T) *recordingRealtime {
	t.Helper()
	r := &recordingRealtime{}
	r.Server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		c, err := websocket.Accept(w, req, &websocket.AcceptOptions{InsecureSkipVerify: true})
		if err != nil {
			return
		}
		defer func() { _ = c.Close(websocket.StatusNormalClosure, "") }()
		ctx := req.Context()
		for {
			_, data, err := c.Read(ctx)
			if err != nil {
				return
			}
			var msg map[string]any
			if json.Unmarshal(data, &msg) != nil {
				continue
			}
			r.mu.Lock()
			r.sent = append(r.sent, msg)
			r.mu.Unlock()
		}
	}))
	return r
}

// awaitSessionWithTools waits for a session.update whose session carries a tool
// named want, returning the tool_choice it was sent with.
func (r *recordingRealtime) awaitSessionWithTools(t *testing.T, want string) string {
	t.Helper()
	deadline := time.Now().Add(3 * time.Second)
	for time.Now().Before(deadline) {
		r.mu.Lock()
		msgs := append([]map[string]any(nil), r.sent...)
		r.mu.Unlock()

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
				spec, ok := tool.(map[string]any)
				if ok && spec["name"] == want {
					choice, _ := session["tool_choice"].(string)
					return choice
				}
			}
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatalf("no session.update advertising tool %q was sent", want)
	return ""
}

// runService starts the realtime service in a pipeline against srv.
func runService(t *testing.T, url string) (*pipeline.Task, chan error) {
	t.Helper()
	svc := realtime.New(realtime.Config{
		APIKey:  "k",
		BaseURL: "ws" + strings.TrimPrefix(url, "http"),
	})
	task := pipeline.NewTask(pipeline.New(svc), pipeline.TaskParams{})
	done := make(chan error, 1)
	go func() { done <- task.Run(context.Background()) }()
	return task, done
}

// TestSetToolsFrameReachesTheSession is the regression test for the defect:
// changing tools mid-conversation must produce a session.update, or the running
// model keeps offering the old toolset.
func TestSetToolsFrameReachesTheSession(t *testing.T) {
	srv := newRecordingRealtime(t)
	defer srv.Close()

	task, done := runService(t, srv.URL)

	task.QueueFrame(frames.NewLLMSetToolsFrame([]frames.Tool{{
		Name:        "get_weather",
		Description: "look up the weather",
		Parameters:  json.RawMessage(`{"type":"object"}`),
	}}))

	if choice := srv.awaitSessionWithTools(t, "get_weather"); choice != string(frames.ToolChoiceAuto) {
		t.Errorf("tool_choice = %q, want auto by default", choice)
	}

	task.StopWhenDone()
	<-done
}

// TestContextFrameSeedsSessionTools checks the toolset configured on the shared
// context reaches the session too, so function calling works without the caller
// having to push a tools frame by hand.
func TestContextFrameSeedsSessionTools(t *testing.T) {
	srv := newRecordingRealtime(t)
	defer srv.Close()

	convo := frames.NewLLMContext("system")
	convo.SetTools([]frames.Tool{{Name: "book_table"}})
	convo.SetToolChoice(frames.ToolChoiceRequired)

	task, done := runService(t, srv.URL)
	task.QueueFrame(frames.NewLLMContextFrame(convo))

	if choice := srv.awaitSessionWithTools(t, "book_table"); choice != string(frames.ToolChoiceRequired) {
		t.Errorf("tool_choice = %q, want required", choice)
	}

	task.StopWhenDone()
	<-done
}
