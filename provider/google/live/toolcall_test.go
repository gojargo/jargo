package live_test

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
	"github.com/gojargo/jargo/processor/aggregators"
	"github.com/gojargo/jargo/provider/google/live"
	"github.com/gojargo/jargo/service/llm"
)

// The Live API asks for tools in one message carrying every call of a turn, and
// takes their results back in a toolResponse. These tests drive that exchange
// against a fake session.

// fakeLive is a Live endpoint that records what the client sends and can speak
// messages to it.
type fakeLive struct {
	*httptest.Server
	mu       sync.Mutex
	received []map[string]any
	toClient chan []byte
}

func newFakeLive(t *testing.T) *fakeLive {
	t.Helper()
	f := &fakeLive{toClient: make(chan []byte, 16)}
	f.Server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		c, err := websocket.Accept(w, r, &websocket.AcceptOptions{InsecureSkipVerify: true})
		if err != nil {
			return
		}
		defer func() { _ = c.Close(websocket.StatusNormalClosure, "") }()
		ctx := r.Context()

		// The session is usable once the setup is acknowledged.
		ready, _ := json.Marshal(map[string]any{"setupComplete": map[string]any{}})
		_ = c.Write(ctx, websocket.MessageText, ready)

		go func() {
			for {
				select {
				case <-ctx.Done():
					return
				case payload := <-f.toClient:
					if c.Write(ctx, websocket.MessageText, payload) != nil {
						return
					}
				}
			}
		}()

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
			f.received = append(f.received, msg)
			f.mu.Unlock()
		}
	}))
	t.Cleanup(f.Close)
	return f
}

// speak sends one server message to the client.
func (f *fakeLive) speak(t *testing.T, raw string) {
	t.Helper()
	f.toClient <- []byte(raw)
}

// received returns a snapshot of what the client has sent.
func (f *fakeLive) messages() []map[string]any {
	f.mu.Lock()
	defer f.mu.Unlock()
	return append([]map[string]any(nil), f.received...)
}

// toolResponses returns every function response the client has sent.
func (f *fakeLive) toolResponses() []map[string]any {
	var out []map[string]any
	for _, m := range f.messages() {
		tr, ok := m["toolResponse"].(map[string]any)
		if !ok {
			continue
		}
		raw, ok := tr["functionResponses"].([]any)
		if !ok {
			continue
		}
		for _, r := range raw {
			if fr, ok := r.(map[string]any); ok {
				out = append(out, fr)
			}
		}
	}
	return out
}

// awaitToolResponses waits for n function responses, or fails.
func (f *fakeLive) awaitToolResponses(t *testing.T, n int) []map[string]any {
	t.Helper()
	deadline := time.Now().Add(3 * time.Second)
	for {
		if got := f.toolResponses(); len(got) >= n {
			return got
		}
		if time.Now().After(deadline) {
			t.Fatalf("the client sent %d function responses, want %d", len(f.toolResponses()), n)
		}
		time.Sleep(10 * time.Millisecond)
	}
}

// awaitToolDeclared waits until a setup message declares a tool, which is how a
// test knows the session carrying it is the one in force.
func (f *fakeLive) awaitToolDeclared(t *testing.T, want string) {
	t.Helper()
	deadline := time.Now().Add(3 * time.Second)
	for time.Now().Before(deadline) {
		for _, m := range f.messages() {
			setup, ok := m["setup"].(map[string]any)
			if !ok {
				continue
			}
			tools, ok := setup["tools"].([]any)
			if !ok {
				continue
			}
			for _, tool := range tools {
				spec, ok := tool.(map[string]any)
				if !ok {
					continue
				}
				decls, ok := spec["functionDeclarations"].([]any)
				if !ok {
					continue
				}
				for _, d := range decls {
					if decl, ok := d.(map[string]any); ok && decl["name"] == want {
						return
					}
				}
			}
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatalf("no session was set up declaring tool %q", want)
}

// toolSession runs the service in a pipeline with the aggregator pair around it,
// which is what carries a tool's result back into the conversation.
func toolSession(t *testing.T, srv *fakeLive, tools []frames.Tool) (*pipeline.Worker, *frames.LLMContext) {
	t.Helper()
	svc := live.New(live.Config{
		APIKey:  "k",
		BaseURL: "ws" + strings.TrimPrefix(srv.URL, "http"),
	})
	convo := frames.NewLLMContext("system")
	convo.SetTools(tools)
	agg := aggregators.New(convo)

	task := pipeline.NewWorker(
		pipeline.New(agg.User(), svc, agg.Assistant()),
		pipeline.WorkerConfig{},
	)
	done := make(chan error, 1)
	go func() { done <- task.Run(context.Background()) }()
	t.Cleanup(func() { task.StopWhenDone(); <-done })

	task.QueueFrame(frames.NewLLMContextFrame(convo))
	// The Live API takes tools only in the setup message, so a conversation
	// bringing them opens the session again. Waiting for the setup that declares
	// them is how a test knows that session is the one now running.
	if len(tools) > 0 {
		srv.awaitToolDeclared(t, tools[0].Name)
	}
	return task, convo
}

// weatherTool carries a handler that records the arguments it was called with.
func weatherTool(ran chan<- string) frames.Tool {
	return frames.Tool{
		Name:        "get_weather",
		Description: "weather",
		Parameters:  json.RawMessage(`{"type":"object"}`),
		Handler: func(ctx context.Context, p llm.FunctionCallParams) error {
			select {
			case ran <- string(p.Arguments):
			default:
			}
			return p.Result(ctx, "sunny", nil)
		},
	}
}

// A conversation that brings tools to a session opened without them reopens it,
// because the Live API takes tools in the setup message and nowhere else.
func TestToolsReachTheSessionSetup(t *testing.T) {
	srv := newFakeLive(t)
	toolSession(t, srv, []frames.Tool{weatherTool(make(chan string, 1))})
	// toolSession waits for the setup declaring the tool, so reaching here is
	// the assertion.
}

// The point of the whole exchange: the model calls a tool and the handler the
// tool carries runs, with the arguments the model wrote.
func TestToolCallRunsTheHandler(t *testing.T) {
	srv := newFakeLive(t)
	ran := make(chan string, 1)
	toolSession(t, srv, []frames.Tool{weatherTool(ran)})

	srv.speak(t, `{"toolCall":{"functionCalls":[
		{"id":"call_1","name":"get_weather","args":{"location":"Paris"}}]}}`)

	select {
	case args := <-ran:
		if !strings.Contains(args, "Paris") {
			t.Errorf("handler got arguments %q, want the ones the model wrote", args)
		}
	case <-time.After(3 * time.Second):
		t.Fatal("the tool the model called never ran")
	}
}

// The result goes back naming the function as well as the call, since that is
// what the response carries.
func TestToolResultReachesTheSession(t *testing.T) {
	srv := newFakeLive(t)
	toolSession(t, srv, []frames.Tool{weatherTool(make(chan string, 1))})

	srv.speak(t, `{"toolCall":{"functionCalls":[
		{"id":"call_1","name":"get_weather","args":{}}]}}`)

	got := srv.awaitToolResponses(t, 1)
	if got[0]["id"] != "call_1" {
		t.Errorf("id = %v, want call_1", got[0]["id"])
	}
	if got[0]["name"] != "get_weather" {
		t.Errorf("name = %v, want the function the call named", got[0]["name"])
	}
	response, ok := got[0]["response"].(map[string]any)
	if !ok || response["value"] != "sunny" {
		t.Errorf("response = %v, want what the handler returned", got[0]["response"])
	}
}

// Every call of a turn arrives in one message, and every one of them runs.
func TestEveryCallInAMessageRuns(t *testing.T) {
	srv := newFakeLive(t)

	var mu sync.Mutex
	var called []string
	tool := func(name string) frames.Tool {
		return frames.Tool{
			Name: name, Parameters: json.RawMessage(`{"type":"object"}`),
			Handler: func(ctx context.Context, p llm.FunctionCallParams) error {
				mu.Lock()
				called = append(called, p.FunctionName)
				mu.Unlock()
				return p.Result(ctx, "ok", nil)
			},
		}
	}
	toolSession(t, srv, []frames.Tool{tool("get_weather"), tool("book_table")})

	srv.speak(t, `{"toolCall":{"functionCalls":[
		{"id":"call_1","name":"get_weather","args":{}},
		{"id":"call_2","name":"book_table","args":{}}]}}`)

	srv.awaitToolResponses(t, 2)
	mu.Lock()
	defer mu.Unlock()
	if len(called) != 2 {
		t.Errorf("ran %v, want both calls of the message", called)
	}
}

// Vertex AI sends no call id, and without one there is nothing to pair a result
// with, so the service supplies one.
func TestACallWithoutAnIDStillAnswers(t *testing.T) {
	srv := newFakeLive(t)
	toolSession(t, srv, []frames.Tool{weatherTool(make(chan string, 1))})

	srv.speak(t, `{"toolCall":{"functionCalls":[{"name":"get_weather","args":{}}]}}`)

	got := srv.awaitToolResponses(t, 1)
	if got[0]["id"] == "" || got[0]["id"] == nil {
		t.Error("the response carried no id, so the model could not pair it with its call")
	}
	if got[0]["name"] != "get_weather" {
		t.Errorf("name = %v, want the function the call named", got[0]["name"])
	}
}

// A result already reported is not reported again: the conversation is handed to
// the service whenever it changes, and every change carries every result it
// holds.
func TestAResultIsSentOnlyOnce(t *testing.T) {
	srv := newFakeLive(t)
	task, convo := toolSession(t, srv, []frames.Tool{weatherTool(make(chan string, 1))})

	srv.speak(t, `{"toolCall":{"functionCalls":[
		{"id":"call_1","name":"get_weather","args":{}}]}}`)
	srv.awaitToolResponses(t, 1)

	convo.AddUserMessage("and tomorrow?")
	task.QueueFrame(frames.NewLLMContextFrame(convo))

	time.Sleep(300 * time.Millisecond)
	if got := srv.toolResponses(); len(got) != 1 {
		t.Errorf("sent %d results, want the one result sent once", len(got))
	}
}
