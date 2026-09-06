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
	"github.com/gojargo/jargo/provider/xai/realtime"
	"github.com/gojargo/jargo/service/llm"
)

// A realtime model asks for a tool by announcing the call and then writing its
// arguments. These tests drive that exchange against a fake session and assert
// on what actually ran and what was sent back.

// toolSession is a Grok session that records what the client sends and can speak
// events to it.
type toolSessionServer struct {
	*httptest.Server
	mu       sync.Mutex
	received []map[string]any
	toClient chan []byte
}

func newToolSessionServer(t *testing.T) *toolSessionServer {
	t.Helper()
	f := &toolSessionServer{toClient: make(chan []byte, 16)}
	f.Server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		c, err := websocket.Accept(w, r, &websocket.AcceptOptions{InsecureSkipVerify: true})
		if err != nil {
			return
		}
		defer func() { _ = c.Close(websocket.StatusNormalClosure, "") }()
		ctx := r.Context()

		// The session accepts configuration only once the conversation is open.
		opened, _ := json.Marshal(map[string]any{"type": "conversation.created"})
		_ = c.Write(ctx, websocket.MessageText, opened)

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
			if msg["type"] == "session.update" {
				// xAI acknowledges the configuration, which is what tells the
				// service the session is ready to be spoken to.
				ack, _ := json.Marshal(map[string]any{"type": "session.updated"})
				if c.Write(ctx, websocket.MessageText, ack) != nil {
					return
				}
			}
		}
	}))
	t.Cleanup(f.Close)
	return f
}

func (f *toolSessionServer) speak(t *testing.T, ev map[string]any) {
	t.Helper()
	payload, err := json.Marshal(ev)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	f.toClient <- payload
}

func (f *toolSessionServer) sentOfType(kind string) []map[string]any {
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

func (f *toolSessionServer) awaitSent(t *testing.T, kind string, n int) []map[string]any {
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

// awaitToolAdvertised waits until the client has told the session about a tool,
// which is how a test knows the conversation carrying it has been handled.
func (f *toolSessionServer) awaitToolAdvertised(t *testing.T, want string) {
	t.Helper()
	deadline := time.Now().Add(3 * time.Second)
	for time.Now().Before(deadline) {
		if sessionAdvertises(f.sentOfType("session.update"), want) {
			return
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatalf("the session was never told about tool %q", want)
}

// announceCall is how the session says the model is calling a tool: the call is
// named first, and its arguments are finished afterwards.
func (f *toolSessionServer) announceCall(t *testing.T, callID, name string) {
	t.Helper()
	f.speak(t, map[string]any{
		"type": "conversation.item.added",
		"item": map[string]any{"type": "function_call", "call_id": callID, "name": name},
	})
}

// finishCallArguments completes a call. Grok names the function here as well as
// on the announcement.
func (f *toolSessionServer) finishCallArguments(t *testing.T, callID, name, args string) {
	t.Helper()
	f.speak(t, map[string]any{
		"type":      "response.function_call_arguments.done",
		"call_id":   callID,
		"name":      name,
		"arguments": args,
	})
}

// toolSession runs the service in a pipeline with the aggregator pair around it,
// which is what carries a tool's result back into the conversation.
func toolSession(t *testing.T, srv *toolSessionServer, tools []frames.Tool) (*pipeline.Worker, *frames.LLMContext) {
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

// The point of the whole exchange: the model calls a tool and the handler the
// tool carries runs.
func TestModelCallRunsTheHandler(t *testing.T) {
	srv := newToolSessionServer(t)

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
	srv.finishCallArguments(t, "call_1", "get_weather", `{"location":"Paris"}`)

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
	srv := newToolSessionServer(t)

	toolSession(t, srv, []frames.Tool{{
		Name:        "get_weather",
		Description: "weather",
		Parameters:  json.RawMessage(`{"type":"object"}`),
		Handler: func(ctx context.Context, p llm.FunctionCallParams) error {
			return p.Result(ctx, "sunny", nil)
		},
	}})

	srv.announceCall(t, "call_1", "get_weather")
	srv.finishCallArguments(t, "call_1", "get_weather", `{}`)

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
	srv := newToolSessionServer(t)

	task, convo := toolSession(t, srv, []frames.Tool{{
		Name:        "get_weather",
		Description: "weather",
		Parameters:  json.RawMessage(`{"type":"object"}`),
		Handler: func(ctx context.Context, p llm.FunctionCallParams) error {
			return p.Result(ctx, "sunny", nil)
		},
	}})

	srv.announceCall(t, "call_1", "get_weather")
	srv.finishCallArguments(t, "call_1", "get_weather", `{}`)
	srv.awaitSent(t, "conversation.item.create", 1)

	convo.AddUserMessage("and tomorrow?")
	task.QueueFrame(frames.NewLLMContextFrame(convo))

	time.Sleep(300 * time.Millisecond)
	if got := srv.sentOfType("conversation.item.create"); len(got) != 1 {
		t.Errorf("sent %d results, want the one result sent once", len(got))
	}
}

// A call the session never announced is not run: the announcement is what says
// the model really asked for it.
func TestAnUnannouncedCallIsNotRun(t *testing.T) {
	srv := newToolSessionServer(t)

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

	srv.finishCallArguments(t, "call_ghost", "get_weather", `{}`)

	select {
	case <-ran:
		t.Error("a call the session never announced was run")
	case <-time.After(300 * time.Millisecond):
	}
}
