package mem0_test

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/provider/mem0"
)

// wire decodes the mem0 request bodies the test inspects.
type searchBody struct {
	Query  string `json:"query"`
	UserID string `json:"user_id"`
	TopK   int    `json:"top_k"`
}

type addBody struct {
	Messages []struct {
		Role    string `json:"role"`
		Content string `json:"content"`
	} `json:"messages"`
	UserID string `json:"user_id"`
}

func TestMemoryRetrievesAndStores(t *testing.T) {
	searched := make(chan searchBody, 1)
	stored := make(chan addBody, 2)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/search":
			var b searchBody
			_ = json.NewDecoder(r.Body).Decode(&b)
			searched <- b
			_ = json.NewEncoder(w).Encode(map[string]any{"results": []map[string]any{
				{"id": "1", "memory": "The user's name is Alex.", "score": 0.9},
				{"id": "2", "memory": "The user prefers tea.", "score": 0.8},
			}})
		case "/memories":
			var b addBody
			_ = json.NewDecoder(r.Body).Decode(&b)
			stored <- b
			_ = json.NewEncoder(w).Encode(map[string]any{"results": []any{}})
		default:
			http.NotFound(w, r)
		}
	}))
	defer srv.Close()

	convo := frames.NewLLMContext("base prompt")
	convo.AddUserMessage("what do you remember about me?")
	m := mem0.NewMemory(mem0.Config{Host: srv.URL, UserID: "u1", SearchLimit: 5})

	task := pipeline.NewTask(pipeline.New(m), pipeline.TaskParams{})
	runDone := make(chan error, 1)
	go func() { runDone <- task.Run(context.Background()) }()
	task.QueueFrame(frames.NewLLMContextFrame(convo))

	select {
	case b := <-searched:
		if b.Query != "what do you remember about me?" || b.UserID != "u1" || b.TopK != 5 {
			t.Fatalf("search request = %+v, want query/user/top_k populated", b)
		}
	case <-time.After(3 * time.Second):
		t.Fatal("mem0 search was not called")
	}

	// The retrieved memories are folded into the system prompt for this turn.
	if !waitFor(2*time.Second, func() bool { return convo.Recall() != "" }) {
		t.Fatal("memories were not injected into the context")
	}
	sys := convo.System()
	if !strings.Contains(sys, "The user's name is Alex.") || !strings.Contains(sys, "The user prefers tea.") {
		t.Fatalf("System() = %q, want it to include the recalled memories", sys)
	}
	if !strings.Contains(sys, "base prompt") {
		t.Fatalf("System() = %q, want it to keep the base prompt", sys)
	}

	select {
	case b := <-stored:
		last := ""
		if len(b.Messages) > 0 {
			last = b.Messages[len(b.Messages)-1].Content
		}
		if b.UserID != "u1" || last != "what do you remember about me?" {
			t.Fatalf("store request = %+v, want the user turn stored under u1", b)
		}
	case <-time.After(3 * time.Second):
		t.Fatal("mem0 store was not called")
	}

	task.StopWhenDone()
	<-runDone
}

func TestMemoryNoResultsClearsRecall(t *testing.T) {
	stored := make(chan struct{}, 1)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/memories" {
			stored <- struct{}{}
		}
		_ = json.NewEncoder(w).Encode(map[string]any{"results": []any{}})
	}))
	defer srv.Close()

	convo := frames.NewLLMContext("base")
	convo.AddUserMessage("hello")
	m := mem0.NewMemory(mem0.Config{Host: srv.URL, UserID: "u1"})

	task := pipeline.NewTask(pipeline.New(m), pipeline.TaskParams{})
	runDone := make(chan error, 1)
	go func() { runDone <- task.Run(context.Background()) }()
	task.QueueFrame(frames.NewLLMContextFrame(convo))

	// Storage runs after injection, so once it fires the recall decision is made.
	select {
	case <-stored:
	case <-time.After(3 * time.Second):
		t.Fatal("mem0 was not called")
	}
	if convo.Recall() != "" {
		t.Fatalf("recall = %q, want empty when search returns no memories", convo.Recall())
	}
	if convo.System() != "base" {
		t.Fatalf("System() = %q, want the untouched base prompt", convo.System())
	}

	task.StopWhenDone()
	<-runDone
}

func waitFor(timeout time.Duration, cond func() bool) bool {
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		if cond() {
			return true
		}
		time.Sleep(5 * time.Millisecond)
	}
	return cond()
}
