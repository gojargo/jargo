package mem0_test

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/provider/mem0"
)

// wire decodes the mem0 request bodies the test inspects.
type searchBody struct {
	Query   string            `json:"query"`
	UserID  string            `json:"user_id"`
	Filters map[string]string `json:"filters"`
	TopK    int               `json:"top_k"`
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
		if b.Query != "what do you remember about me?" || b.Filters["user_id"] != "u1" || b.TopK != 5 {
			t.Fatalf("search request = %+v, want query/filters.user_id/top_k populated", b)
		}
		// The entity scope belongs under "filters"; search rejects it top-level.
		if b.UserID != "" {
			t.Fatalf("search request carried a top-level user_id = %q, want it only in filters", b.UserID)
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

// countingServer answers every search and records the queries it was asked, so a
// test can tell how many times retrieval actually ran.
func countingServer(t *testing.T, delay time.Duration) (*httptest.Server, func() []string) {
	t.Helper()
	var mu sync.Mutex
	var queries []string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/search" {
			var b searchBody
			_ = json.NewDecoder(r.Body).Decode(&b)
			mu.Lock()
			queries = append(queries, b.Query)
			mu.Unlock()
			if delay > 0 {
				select {
				case <-time.After(delay):
				case <-r.Context().Done():
					return
				}
			}
		}
		_ = json.NewEncoder(w).Encode(map[string]any{"results": []any{}})
	}))
	return srv, func() []string {
		mu.Lock()
		defer mu.Unlock()
		return slices.Clone(queries)
	}
}

// searchQueries returns the non-prewarm queries seen so far.
func searchQueries(all []string) []string {
	var out []string
	for _, q := range all {
		if q != "hello" {
			out = append(out, q)
		}
	}
	return out
}

// A tool-calling turn replays the same context once per round trip. Retrieval is
// keyed on the user message, so the replays must not each cost a search.
func TestMemorySearchesOncePerUserMessage(t *testing.T) {
	srv, queries := countingServer(t, 0)
	defer srv.Close()

	convo := frames.NewLLMContext("base")
	convo.AddUserMessage("where did I put my keys?")
	m := mem0.NewMemory(mem0.Config{Host: srv.URL, UserID: "u1"})

	task := pipeline.NewTask(pipeline.New(m), pipeline.TaskParams{})
	runDone := make(chan error, 1)
	go func() { runDone <- task.Run(context.Background()) }()

	for range 3 {
		task.QueueFrame(frames.NewLLMContextFrame(convo))
	}
	if !waitFor(3*time.Second, func() bool { return len(searchQueries(queries())) >= 1 }) {
		t.Fatal("mem0 search was never called")
	}

	// A second user message is a new query and must search again.
	convo.AddUserMessage("and my glasses?")
	task.QueueFrame(frames.NewLLMContextFrame(convo))
	if !waitFor(3*time.Second, func() bool { return len(searchQueries(queries())) >= 2 }) {
		t.Fatal("the second user message did not trigger a search")
	}

	task.StopWhenDone()
	<-runDone

	got := searchQueries(queries())
	want := []string{"where did I put my keys?", "and my glasses?"}
	if !slices.Equal(got, want) {
		t.Errorf("searched for %q, want %q — one search per user message", got, want)
	}
}

func TestMemoryPrewarmsTheSearchPath(t *testing.T) {
	srv, queries := countingServer(t, 0)
	defer srv.Close()

	m := mem0.NewMemory(mem0.Config{Host: srv.URL, UserID: "u1", Prewarm: true})
	task := pipeline.NewTask(pipeline.New(m), pipeline.TaskParams{})
	runDone := make(chan error, 1)
	go func() { runDone <- task.Run(context.Background()) }()

	// StartFrame alone must warm the path, before any user message exists.
	if !waitFor(3*time.Second, func() bool { return len(queries()) > 0 }) {
		t.Fatal("no search was issued on start, so the path was not prewarmed")
	}

	task.StopWhenDone()
	<-runDone
}

func TestMemoryPrewarmIsOptIn(t *testing.T) {
	srv, queries := countingServer(t, 0)
	defer srv.Close()

	m := mem0.NewMemory(mem0.Config{Host: srv.URL, UserID: "u1"})
	task := pipeline.NewTask(pipeline.New(m), pipeline.TaskParams{})
	runDone := make(chan error, 1)
	go func() { runDone <- task.Run(context.Background()) }()

	time.Sleep(200 * time.Millisecond)
	if got := queries(); len(got) != 0 {
		t.Errorf("searched %q on start, want nothing without Prewarm", got)
	}

	task.StopWhenDone()
	<-runDone
}

// A slow server must not hold the turn past SearchTimeout: the reply goes out
// without memories rather than late with them.
func TestMemorySearchTimeoutReleasesTheTurn(t *testing.T) {
	srv, _ := countingServer(t, 2*time.Second)
	defer srv.Close()

	convo := frames.NewLLMContext("base")
	convo.AddUserMessage("hello there")
	m := mem0.NewMemory(mem0.Config{
		Host: srv.URL, UserID: "u1",
		Timeout:       10 * time.Second,
		SearchTimeout: 100 * time.Millisecond,
	})

	downstream := newFrameSink()
	task := pipeline.NewTask(pipeline.New(m, downstream), pipeline.TaskParams{})
	runDone := make(chan error, 1)
	go func() { runDone <- task.Run(context.Background()) }()

	start := time.Now()
	task.QueueFrame(frames.NewLLMContextFrame(convo))
	if !waitFor(3*time.Second, func() bool { return downstream.sawContext() }) {
		t.Fatal("the context frame never reached the LLM")
	}
	// Well under the server's 2s, so the deadline and not the server ended it.
	if elapsed := time.Since(start); elapsed > time.Second {
		t.Errorf("context frame took %v to pass, want it released near the 100ms SearchTimeout", elapsed)
	}

	task.StopWhenDone()
	<-runDone
}

// frameSink records whether a context frame made it downstream of memory.
type frameSink struct {
	*processor.Base
	mu  sync.Mutex
	got bool
}

func newFrameSink() *frameSink {
	s := &frameSink{}
	s.Base = processor.New("Sink", s)
	return s
}

func (s *frameSink) ProcessFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	if err := s.Base.ProcessFrame(ctx, f, dir); err != nil {
		return err
	}
	if _, ok := f.(*frames.LLMContextFrame); ok {
		s.mu.Lock()
		s.got = true
		s.mu.Unlock()
	}
	return s.PushFrame(ctx, f, dir)
}

func (s *frameSink) sawContext() bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.got
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
