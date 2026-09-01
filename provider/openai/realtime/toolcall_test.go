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
	"github.com/gojargo/jargo/processor/aggregators"
	"github.com/gojargo/jargo/provider/openai/realtime"
	"github.com/gojargo/jargo/service/llm"
)

// A realtime model asks for a tool by announcing the call and then writing its
// arguments. These tests drive that exchange against a fake session and assert
// on what actually ran and what was sent back.

// fakeSession is a Realtime endpoint that records what the client sends and can
// speak events to it.
type fakeSession struct {
	*httptest.Server
	mu       sync.Mutex
	received []map[string]any
	toClient chan []byte
}

func newFakeSession(t *testing.T) *fakeSession {
	t.Helper()
	f := &fakeSession{toClient: make(chan []byte, 16)}
	f.Server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		c, err := websocket.Accept(w, req, &websocket.AcceptOptions{InsecureSkipVerify: true})
		if err != nil {
			return
		}
		defer func() { _ = c.Close(websocket.StatusNormalClosure, "") }()
		ctx := req.Context()

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

// speak sends one server event to the client.
func (f *fakeSession) speak(t *testing.T, ev map[string]any) {
	t.Helper()
	payload, err := json.Marshal(ev)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	f.toClient <- payload
}

// sentOfType returns every message of the given type the client has sent.
func (f *fakeSession) sentOfType(kind string) []map[string]any {
	f.mu.Lock()
	defer f.mu.Unlock()
	var out []map[string]any
	for _, m := range f.received {
		if m["type"] == kind {
			out = append(out, m)
		}
	}
	return out
}

// awaitSent waits for n messages of a type, or fails.
func (f *fakeSession) awaitSent(t *testing.T, kind string, n int) []map[string]any {
	t.Helper()
	deadline := time.Now().Add(3 * time.Second)
	for {
		if got := f.sentOfType(kind); len(got) >= n {
			return got
		}
		if time.Now().After(deadline) {
			t.Fatalf("the client sent %d %q messages, want %d", len(f.sentOfType(kind)), kind, n)
		}
		time.Sleep(10 * time.Millisecond)
	}
}

// announceCall is how the session says the model is calling a tool: the call is
// named first, and its arguments are finished afterwards.
func (f *fakeSession) announceCall(t *testing.T, callID, name string) {
	t.Helper()
	f.speak(t, map[string]any{
		"type": "conversation.item.added",
		"item": map[string]any{"type": "function_call", "call_id": callID, "name": name},
	})
}

func (f *fakeSession) finishCallArguments(t *testing.T, callID, args string) {
	t.Helper()
	f.speak(t, map[string]any{
		"type":      "response.function_call_arguments.done",
		"call_id":   callID,
		"arguments": args,
	})
}

// toolSession runs the service in a pipeline with the aggregator pair around it,
// which is what carries a tool's result back into the conversation.
func toolSession(t *testing.T, srv *fakeSession, tools []frames.Tool) (*pipeline.Worker, *frames.LLMContext) {
	t.Helper()
	svc := realtime.New(realtime.Config{
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
	// A tool call is answered into the conversation, so the service has to have
	// been given one before the session may call anything. The session update
	// carrying the toolset is what says the conversation arrived: it is sent
	// from the same handling.
	if len(tools) > 0 {
		srv.awaitToolAdvertised(t, tools[0].Name)
	}
	return task, convo
}

// awaitToolAdvertised waits until the client has told the session about a tool,
// which is how a test knows the conversation carrying it has been handled.
func (f *fakeSession) awaitToolAdvertised(t *testing.T, want string) {
	t.Helper()
	deadline := time.Now().Add(3 * time.Second)
	for time.Now().Before(deadline) {
		for _, m := range f.sentOfType("session.update") {
			session, ok := m["session"].(map[string]any)
			if !ok {
				continue
			}
			advertised, ok := session["tools"].([]any)
			if !ok {
				continue
			}
			for _, tool := range advertised {
				if spec, ok := tool.(map[string]any); ok && spec["name"] == want {
					return
				}
			}
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatalf("the session was never told about tool %q", want)
}

// The point of the whole exchange: the model calls a tool and the handler the
// tool carries runs.
func TestModelCallRunsTheHandler(t *testing.T) {
	srv := newFakeSession(t)

	ran := make(chan string, 1)
	toolSession(t, srv, []frames.Tool{{
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
	}})

	srv.announceCall(t, "call_1", "get_weather")
	srv.finishCallArguments(t, "call_1", `{"location":"Paris"}`)

	select {
	case args := <-ran:
		if !strings.Contains(args, "Paris") {
			t.Errorf("handler got arguments %q, want the ones the model wrote", args)
		}
	case <-time.After(3 * time.Second):
		t.Fatal("the tool the model called never ran")
	}
}

// The result has to reach the model, and the model has to be asked to speak: it
// is generating from audio otherwise, so nothing else would prompt the answer.
func TestToolResultReachesTheSessionAndPromptsAReply(t *testing.T) {
	srv := newFakeSession(t)

	toolSession(t, srv, []frames.Tool{{
		Name:        "get_weather",
		Description: "weather",
		Parameters:  json.RawMessage(`{"type":"object"}`),
		Handler: func(ctx context.Context, p llm.FunctionCallParams) error {
			return p.Result(ctx, "sunny", nil)
		},
	}})

	srv.announceCall(t, "call_1", "get_weather")
	srv.finishCallArguments(t, "call_1", `{}`)

	items := srv.awaitSent(t, "conversation.item.create", 1)
	item, _ := items[0]["item"].(map[string]any)
	if item["type"] != "function_call_output" {
		t.Errorf("item = %v, want a function_call_output", item)
	}
	if item["call_id"] != "call_1" {
		t.Errorf("call_id = %v, want call_1", item["call_id"])
	}
	if item["output"] != "sunny" {
		t.Errorf("output = %v, want what the handler returned", item["output"])
	}

	srv.awaitSent(t, "response.create", 1)
}

// A result already reported is not reported again. The conversation is handed to
// the service whenever it changes, so without this every later change would
// resend every result it holds.
func TestAResultIsSentOnlyOnce(t *testing.T) {
	srv := newFakeSession(t)

	task, convo := toolSession(t, srv, []frames.Tool{{
		Name:        "get_weather",
		Description: "weather",
		Parameters:  json.RawMessage(`{"type":"object"}`),
		Handler: func(ctx context.Context, p llm.FunctionCallParams) error {
			return p.Result(ctx, "sunny", nil)
		},
	}})

	srv.announceCall(t, "call_1", "get_weather")
	srv.finishCallArguments(t, "call_1", `{}`)
	srv.awaitSent(t, "conversation.item.create", 1)

	// The conversation changes again for a reason that has nothing to do with
	// the tool call.
	convo.AddUserMessage("and tomorrow?")
	task.QueueFrame(frames.NewLLMContextFrame(convo))

	time.Sleep(300 * time.Millisecond)
	if got := srv.sentOfType("conversation.item.create"); len(got) != 1 {
		t.Errorf("sent %d results, want the one result sent once", len(got))
	}
}

// A call the session never announced has no name to run, so it is reported
// rather than guessed at.
func TestAnUnannouncedCallIsNotRun(t *testing.T) {
	srv := newFakeSession(t)

	ran := make(chan struct{}, 1)
	toolSession(t, srv, []frames.Tool{{
		Name:       "get_weather",
		Parameters: json.RawMessage(`{"type":"object"}`),
		Handler: func(ctx context.Context, p llm.FunctionCallParams) error {
			select {
			case ran <- struct{}{}:
			default:
			}
			return p.Result(ctx, "sunny", nil)
		},
	}})

	// Arguments for a call that was never announced.
	srv.finishCallArguments(t, "call_ghost", `{}`)

	select {
	case <-ran:
		t.Error("a call the session never announced was run")
	case <-time.After(300 * time.Millisecond):
	}
}

// The arguments event can repeat; the call must not. It is taken out of the
// pending set before it runs, which is what makes the second event a no-op.
func TestARepeatedArgumentsEventRunsTheCallOnce(t *testing.T) {
	srv := newFakeSession(t)

	var mu sync.Mutex
	runs := 0
	toolSession(t, srv, []frames.Tool{{
		Name:       "get_weather",
		Parameters: json.RawMessage(`{"type":"object"}`),
		Handler: func(ctx context.Context, p llm.FunctionCallParams) error {
			mu.Lock()
			runs++
			mu.Unlock()
			return p.Result(ctx, "sunny", nil)
		},
	}})

	srv.announceCall(t, "call_1", "get_weather")
	srv.finishCallArguments(t, "call_1", `{}`)
	srv.finishCallArguments(t, "call_1", `{}`)

	srv.awaitSent(t, "conversation.item.create", 1)
	time.Sleep(200 * time.Millisecond)

	mu.Lock()
	defer mu.Unlock()
	if runs != 1 {
		t.Errorf("the tool ran %d times, want once", runs)
	}
}
