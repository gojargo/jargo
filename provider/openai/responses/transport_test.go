package responses

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"sync"
	"testing"

	"github.com/coder/websocket"
	"github.com/gojargo/jargo/frames"
)

// recordingSink collects what a turn produced.
type recordingSink struct {
	text  strings.Builder
	calls []frames.ToolCall
}

func (s *recordingSink) Text(text string) error {
	s.text.WriteString(text)
	return nil
}

func (s *recordingSink) Tool(call frames.ToolCall) error {
	s.calls = append(s.calls, call)
	return nil
}

// feed folds a sequence of events into a fresh state machine.
func feed(t *testing.T, sink *recordingSink, events ...string) (*streamState, error) {
	t.Helper()
	state := newStreamState(sink)
	for _, raw := range events {
		ev, ok := decodeEvent([]byte(raw))
		if !ok {
			t.Fatalf("test event is not valid JSON: %s", raw)
		}
		done, err := state.handle(ev)
		if err != nil {
			return state, err
		}
		if done {
			return state, nil
		}
	}
	return state, nil
}

// TestStreamStateText checks text deltas reach the sink and the response id and
// usage are captured from the lifecycle events.
func TestStreamStateText(t *testing.T) {
	sink := &recordingSink{}
	state, err := feed(t, sink,
		`{"type":"response.created","response":{"id":"resp_1"}}`,
		`{"type":"response.output_text.delta","delta":"Hello"}`,
		`{"type":"response.output_text.delta","delta":" there"}`,
		`{"type":"response.completed","response":{"id":"resp_1","usage":`+
			`{"input_tokens":10,"output_tokens":5,"total_tokens":15,`+
			`"input_tokens_details":{"cached_tokens":4}}}}`,
	)
	if err != nil {
		t.Fatalf("feed: %v", err)
	}
	if got := sink.text.String(); got != "Hello there" {
		t.Errorf("text = %q, want the concatenated deltas", got)
	}
	if state.responseID != "resp_1" {
		t.Errorf("responseID = %q, want resp_1", state.responseID)
	}
	if state.usage == nil {
		t.Fatal("usage not captured")
	}
	want := frames.LLMTokenUsage{
		PromptTokens: 10, CompletionTokens: 5, TotalTokens: 15,
		CacheReadTokens: new(int64(4)),
	}
	if got := state.usage.tokenUsage(); !reflect.DeepEqual(got, want) {
		t.Errorf("tokenUsage = %+v, want %+v", got, want)
	}
	if state.outputItems() != 1 {
		t.Errorf("outputItems = %d, want 1 for a text-only response", state.outputItems())
	}
}

// TestStreamStateToolCalls checks function calls are assembled from their
// fragments and reported in output order once the turn ends.
func TestStreamStateToolCalls(t *testing.T) {
	sink := &recordingSink{}
	state, err := feed(t, sink,
		`{"type":"response.created","response":{"id":"resp_2"}}`,
		`{"type":"response.output_item.added","output_index":0,`+
			`"item":{"type":"function_call","call_id":"call_a","name":"first"}}`,
		`{"type":"response.output_item.added","output_index":1,`+
			`"item":{"type":"function_call","call_id":"call_b","name":"second"}}`,
		// Arguments arrive interleaved across the two calls.
		`{"type":"response.function_call_arguments.delta","output_index":1,"delta":"{\"y\":"}`,
		`{"type":"response.function_call_arguments.delta","output_index":0,"delta":"{\"x\":1}"}`,
		`{"type":"response.function_call_arguments.delta","output_index":1,"delta":"2}"}`,
		`{"type":"response.completed","response":{"id":"resp_2"}}`,
	)
	if err != nil {
		t.Fatalf("feed: %v", err)
	}
	if len(state.calls) != 0 {
		t.Errorf("calls left pending = %+v, want them all flushed", state.calls)
	}
	if len(sink.calls) != 2 {
		t.Fatalf("calls = %+v, want two", sink.calls)
	}
	if sink.calls[0].Name != "first" || string(sink.calls[0].Args) != `{"x":1}` {
		t.Errorf("call 0 = %+v, want first with its own arguments", sink.calls[0])
	}
	if sink.calls[1].Name != "second" || string(sink.calls[1].Args) != `{"y":2}` {
		t.Errorf("call 1 = %+v, want second with its own arguments", sink.calls[1])
	}
	if state.outputItems() != 2 {
		t.Errorf("outputItems = %d, want one per call", state.outputItems())
	}
}

// TestStreamStateArgsDoneSupersedes checks the completed-arguments event
// replaces the accumulated fragments rather than being appended to them, which
// would produce invalid JSON.
func TestStreamStateArgsDoneSupersedes(t *testing.T) {
	sink := &recordingSink{}
	if _, err := feed(t, sink,
		`{"type":"response.output_item.added","output_index":0,"item":{"type":"function_call","call_id":"c","name":"f"}}`,
		`{"type":"response.function_call_arguments.delta","output_index":0,"delta":"{\"a\":"}`,
		`{"type":"response.function_call_arguments.done","output_index":0,"arguments":"{\"a\":1}"}`,
		`{"type":"response.completed","response":{"id":"r"}}`,
	); err != nil {
		t.Fatalf("feed: %v", err)
	}
	if len(sink.calls) != 1 || string(sink.calls[0].Args) != `{"a":1}` {
		t.Errorf("calls = %+v, want the completed arguments alone", sink.calls)
	}
}

// TestStreamStateEmptyArguments checks a call with no arguments is reported as
// an empty object, which downstream JSON unmarshalling requires.
func TestStreamStateEmptyArguments(t *testing.T) {
	sink := &recordingSink{}
	if _, err := feed(t, sink,
		`{"type":"response.output_item.added","output_index":0,"item":{"type":"function_call","call_id":"c","name":"now"}}`,
		`{"type":"response.completed","response":{"id":"r"}}`,
	); err != nil {
		t.Fatalf("feed: %v", err)
	}
	if len(sink.calls) != 1 || string(sink.calls[0].Args) != "{}" {
		t.Errorf("calls = %+v, want empty-object arguments", sink.calls)
	}
}

// TestStreamStateFailures checks each way a turn can fail is surfaced.
func TestStreamStateFailures(t *testing.T) {
	cases := []struct {
		name  string
		event string
		want  string
	}{
		{
			"response failed with a message",
			`{"type":"response.failed","response":{"id":"r","status":"failed","error":{"message":"model overloaded"}}}`,
			"model overloaded",
		},
		{
			"response failed without one",
			`{"type":"response.failed","response":{"id":"r","status":"failed"}}`,
			"failed",
		},
		{
			"stream error",
			`{"type":"error","message":"bad request"}`,
			"bad request",
		},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			_, err := feed(t, &recordingSink{}, c.event)
			if err == nil {
				t.Fatal("feed() = nil error, want the failure surfaced")
			}
			if !errors.Is(err, errServer) {
				t.Errorf("error = %v, want it to wrap errServer", err)
			}
			if !strings.Contains(err.Error(), c.want) {
				t.Errorf("error = %v, want it to mention %q", err, c.want)
			}
		})
	}
}

// TestStreamStateIncomplete checks a response cut short still delivers what the
// model produced rather than being treated as a failure.
func TestStreamStateIncomplete(t *testing.T) {
	sink := &recordingSink{}
	if _, err := feed(t, sink,
		`{"type":"response.output_text.delta","delta":"partial"}`,
		`{"type":"response.incomplete","response":{"id":"r","incomplete_details":{"reason":"max_output_tokens"}}}`,
	); err != nil {
		t.Fatalf("feed: %v", err)
	}
	if sink.text.String() != "partial" {
		t.Errorf("text = %q, want what was produced before the cut", sink.text.String())
	}
}

// TestHTTPGenerate drives the HTTP service end to end against a fake endpoint.
func TestHTTPGenerate(t *testing.T) {
	var gotPath, gotAuth string
	var gotBody map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotPath = r.URL.Path
		gotAuth = r.Header.Get("Authorization")
		_ = json.NewDecoder(r.Body).Decode(&gotBody)
		w.Header().Set("Content-Type", "text/event-stream")
		for _, ev := range []string{
			`{"type":"response.created","response":{"id":"resp_1"}}`,
			`{"type":"response.output_text.delta","delta":"hi"}`,
			`{"type":"response.completed","response":{"id":"resp_1"}}`,
		} {
			_, _ = w.Write([]byte("data: " + ev + "\n\n"))
		}
	}))
	defer srv.Close()

	svc := NewHTTPLLM(Config{APIKey: "test-key", BaseURL: srv.URL})
	svc.http = srv.Client()

	convo := frames.NewLLMContext("be brief")
	convo.AddUserMessage("hello")

	var text strings.Builder
	if err := svc.Generate(context.Background(), convo, func(s string) error {
		text.WriteString(s)
		return nil
	}); err != nil {
		t.Fatalf("Generate: %v", err)
	}

	if text.String() != "hi" {
		t.Errorf("text = %q, want the streamed delta", text.String())
	}
	if gotPath != "/responses" {
		t.Errorf("path = %q, want /responses", gotPath)
	}
	if gotAuth != "Bearer test-key" {
		t.Errorf("Authorization = %q, want the Bearer key", gotAuth)
	}
	if gotBody["instructions"] != "be brief" {
		t.Errorf("instructions = %v, want the system prompt", gotBody["instructions"])
	}
	if _, ok := gotBody["previous_response_id"]; ok {
		t.Error("the HTTP service sent previous_response_id, which needs store=true over HTTP")
	}
}

// TestHTTPErrorStatus checks a non-200 is reported rather than parsed.
func TestHTTPErrorStatus(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusUnauthorized)
		_, _ = w.Write([]byte("bad key"))
	}))
	defer srv.Close()

	svc := NewHTTPLLM(Config{APIKey: "k", BaseURL: srv.URL})
	svc.http = srv.Client()

	err := svc.Generate(context.Background(), frames.NewLLMContext(""), func(string) error { return nil })
	if err == nil {
		t.Fatal("Generate() = nil error, want the status reported")
	}
	if !errors.Is(err, errStatus) || !strings.Contains(err.Error(), "bad key") {
		t.Errorf("error = %v, want it to wrap errStatus and carry the body", err)
	}
}

// wsTurn is one turn a fake Responses WebSocket served.
type wsTurn struct {
	request map[string]any
}

// fakeResponsesWS is a WebSocket standing in for the Responses endpoint. It
// records each response.create it receives and replies with a scripted turn.
type fakeResponsesWS struct {
	*httptest.Server
	mu    sync.Mutex
	turns []wsTurn
}

// newFakeResponsesWS starts a server that answers the Nth request with the Nth
// scripted event list.
func newFakeResponsesWS(t *testing.T, scripts [][]string) *fakeResponsesWS {
	t.Helper()
	f := &fakeResponsesWS{}
	f.Server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		c, err := websocket.Accept(w, r, &websocket.AcceptOptions{InsecureSkipVerify: true})
		if err != nil {
			return
		}
		defer func() { _ = c.Close(websocket.StatusNormalClosure, "") }()
		ctx := r.Context()

		for {
			_, data, err := c.Read(ctx)
			if err != nil {
				return
			}
			var msg struct {
				Response map[string]any `json:"response"`
			}
			if json.Unmarshal(data, &msg) != nil {
				continue
			}
			f.mu.Lock()
			n := len(f.turns)
			f.turns = append(f.turns, wsTurn{request: msg.Response})
			f.mu.Unlock()

			if n >= len(scripts) {
				return
			}
			for _, ev := range scripts[n] {
				if c.Write(ctx, websocket.MessageText, []byte(ev)) != nil {
					return
				}
			}
		}
	}))
	t.Cleanup(f.Close)
	return f
}

// wsURL is the server's WebSocket address.
func (f *fakeResponsesWS) wsURL() string { return "ws" + strings.TrimPrefix(f.URL, "http") }

// turn returns the nth recorded request.
func (f *fakeResponsesWS) turn(n int) map[string]any {
	f.mu.Lock()
	defer f.mu.Unlock()
	if n >= len(f.turns) {
		return nil
	}
	return f.turns[n].request
}

// TestWSIncrementalContext is the point of the WebSocket service: after a first
// turn sends the whole conversation, the second sends only what is new and lets
// the server recall the rest by response id.
func TestWSIncrementalContext(t *testing.T) {
	srv := newFakeResponsesWS(t, [][]string{
		{
			`{"type":"response.created","response":{"id":"resp_1"}}`,
			`{"type":"response.output_text.delta","delta":"hi there"}`,
			`{"type":"response.completed","response":{"id":"resp_1"}}`,
		},
		{
			`{"type":"response.created","response":{"id":"resp_2"}}`,
			`{"type":"response.output_text.delta","delta":"sunny"}`,
			`{"type":"response.completed","response":{"id":"resp_2"}}`,
		},
	})

	svc := NewLLM(Config{APIKey: "k", WSURL: srv.wsURL()})
	t.Cleanup(svc.disconnect)

	convo := frames.NewLLMContext("be brief")
	convo.AddUserMessage("hello")
	discard := func(string) error { return nil }

	if err := svc.Generate(context.Background(), convo, discard); err != nil {
		t.Fatalf("first Generate: %v", err)
	}
	first := srv.turn(0)
	if input, _ := first["input"].([]any); len(input) != 1 {
		t.Fatalf("first turn input = %v, want the whole conversation", first["input"])
	}
	if _, ok := first["previous_response_id"]; ok {
		t.Error("the first turn continued from a response that does not exist yet")
	}

	// The turn lands in the context, then the user speaks again.
	convo.AddAssistantMessage("hi there")
	convo.AddUserMessage("weather?")

	if err := svc.Generate(context.Background(), convo, discard); err != nil {
		t.Fatalf("second Generate: %v", err)
	}
	second := srv.turn(1)
	if got := second["previous_response_id"]; got != "resp_1" {
		t.Errorf("previous_response_id = %v, want resp_1", got)
	}
	input, _ := second["input"].([]any)
	if len(input) != 1 {
		t.Fatalf("second turn input = %v, want only the new message", second["input"])
	}
	item, _ := input[0].(map[string]any)
	if item["content"] != "weather?" {
		t.Errorf("second turn sent %v, want only the new user message", item)
	}
}

// TestWSFullContextWhenHistoryChanges checks the service falls back to sending
// everything when the conversation was rewritten behind its back, rather than
// continuing from a server copy that no longer matches.
func TestWSFullContextWhenHistoryChanges(t *testing.T) {
	srv := newFakeResponsesWS(t, [][]string{
		{
			`{"type":"response.created","response":{"id":"resp_1"}}`,
			`{"type":"response.output_text.delta","delta":"a"}`,
			`{"type":"response.completed","response":{"id":"resp_1"}}`,
		},
		{
			`{"type":"response.created","response":{"id":"resp_2"}}`,
			`{"type":"response.output_text.delta","delta":"b"}`,
			`{"type":"response.completed","response":{"id":"resp_2"}}`,
		},
	})

	svc := NewLLM(Config{APIKey: "k", WSURL: srv.wsURL()})
	t.Cleanup(svc.disconnect)

	convo := frames.NewLLMContext("")
	convo.AddUserMessage("hello")
	discard := func(string) error { return nil }
	if err := svc.Generate(context.Background(), convo, discard); err != nil {
		t.Fatalf("first Generate: %v", err)
	}

	// Rewrite the history: the prefix no longer matches what was sent.
	convo.SetMessages([]frames.Message{
		{Role: frames.RoleUser, Text: "completely different"},
		{Role: frames.RoleAssistant, Text: "a"},
		{Role: frames.RoleUser, Text: "and more"},
	})
	if err := svc.Generate(context.Background(), convo, discard); err != nil {
		t.Fatalf("second Generate: %v", err)
	}

	second := srv.turn(1)
	if _, ok := second["previous_response_id"]; ok {
		t.Error("continued from the previous response despite the history changing")
	}
	if input, _ := second["input"].([]any); len(input) != 3 {
		t.Errorf("second turn input = %v, want the whole rewritten conversation", second["input"])
	}
}

// TestWSRetriesOnStaleConnection checks the two recoverable server errors drop
// the connection and retry with the full context, rather than failing the turn.
func TestWSRetriesOnStaleConnection(t *testing.T) {
	for _, code := range []string{"previous_response_not_found", "connection_expired"} {
		t.Run(code, func(t *testing.T) {
			srv := newFakeResponsesWS(t, [][]string{
				{`{"type":"error","code":"` + code + `","message":"stale"}`},
				{
					`{"type":"response.created","response":{"id":"resp_2"}}`,
					`{"type":"response.output_text.delta","delta":"recovered"}`,
					`{"type":"response.completed","response":{"id":"resp_2"}}`,
				},
			})

			svc := NewLLM(Config{APIKey: "k", WSURL: srv.wsURL()})
			t.Cleanup(svc.disconnect)

			convo := frames.NewLLMContext("")
			convo.AddUserMessage("hello")

			var text strings.Builder
			if err := svc.Generate(context.Background(), convo, func(s string) error {
				text.WriteString(s)
				return nil
			}); err != nil {
				t.Fatalf("Generate: %v", err)
			}
			if text.String() != "recovered" {
				t.Errorf("text = %q, want the retry's output", text.String())
			}
		})
	}
}

// TestIsRecoverable checks which failures trigger a retry. An interruption must
// not, or a barge-in would produce a second generation.
func TestIsRecoverable(t *testing.T) {
	cases := []struct {
		name string
		err  error
		want bool
	}{
		{"no error", nil, false},
		{"previous response gone", errPreviousResponseGone, true},
		{"connection expired", errConnectionExpired, true},
		{"canceled", context.Canceled, false},
		{"server error", errServer, false},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := isRecoverable(c.err); got != c.want {
				t.Errorf("isRecoverable(%v) = %v, want %v", c.err, got, c.want)
			}
		})
	}
}

// TestClassifyStreamError checks only the two connection-scoped codes are
// treated as recoverable; anything else flows on as a normal stream error.
func TestClassifyStreamError(t *testing.T) {
	gone := classifyStreamError(event{Type: evtError, Code: "previous_response_not_found"})
	if !errors.Is(gone, errPreviousResponseGone) {
		t.Errorf("previous_response_not_found = %v, want errPreviousResponseGone", gone)
	}
	expired := classifyStreamError(event{Type: evtError, Code: "connection_expired"})
	if !errors.Is(expired, errConnectionExpired) {
		t.Errorf("connection_expired = %v, want errConnectionExpired", expired)
	}
	if err := classifyStreamError(event{Type: evtError, Code: "invalid_request"}); err != nil {
		t.Errorf("invalid_request = %v, want nil so it flows on as a stream error", err)
	}
	if err := classifyStreamError(event{Type: evtTextDelta}); err != nil {
		t.Errorf("non-error event = %v, want nil", err)
	}
}

// interruptibleWS is a fake endpoint whose first turn starts generating and
// never finishes on its own, so the client has to cancel it. Reads and writes
// share the handler goroutine, which returns to reading immediately, so a
// cancel arriving mid-generation is still processed.
type interruptibleWS struct {
	*httptest.Server
	started  chan struct{}
	mu       sync.Mutex
	canceled string
	requests []map[string]any
}

func newInterruptibleWS(t *testing.T) *interruptibleWS {
	t.Helper()
	f := &interruptibleWS{started: make(chan struct{}, 4)}
	f.Server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		c, err := websocket.Accept(w, r, &websocket.AcceptOptions{InsecureSkipVerify: true})
		if err != nil {
			return
		}
		defer func() { _ = c.Close(websocket.StatusNormalClosure, "") }()
		ctx := r.Context()

		write := func(ev string) bool { return c.Write(ctx, websocket.MessageText, []byte(ev)) == nil }

		for {
			_, data, err := c.Read(ctx)
			if err != nil {
				return
			}
			var msg struct {
				Type       string         `json:"type"`
				ResponseID string         `json:"response_id"`
				Response   map[string]any `json:"response"`
			}
			if json.Unmarshal(data, &msg) != nil {
				continue
			}

			if msg.Type == "response.cancel" {
				f.mu.Lock()
				f.canceled = msg.ResponseID
				f.mu.Unlock()
				// The API terminates a canceled response rather than dropping it.
				if !write(`{"type":"response.incomplete","response":{"id":"` + msg.ResponseID + `"}}`) {
					return
				}
				continue
			}

			f.mu.Lock()
			f.requests = append(f.requests, msg.Response)
			n := len(f.requests)
			f.mu.Unlock()

			if n == 1 {
				// Start generating, then stall: only a cancel ends this turn.
				if !write(`{"type":"response.created","response":{"id":"resp_1"}}`) ||
					!write(`{"type":"response.output_text.delta","delta":"thinking"}`) {
					return
				}
				f.started <- struct{}{}
				continue
			}
			for _, ev := range []string{
				`{"type":"response.created","response":{"id":"resp_2"}}`,
				`{"type":"response.output_text.delta","delta":"second"}`,
				`{"type":"response.completed","response":{"id":"resp_2"}}`,
			} {
				if !write(ev) {
					return
				}
			}
		}
	}))
	t.Cleanup(f.Close)
	return f
}

// TestWSInterruptionCleansConnection checks a barge-in cancels the response
// server-side and drains its remaining events, so the next turn reads its own
// output rather than the interrupted one's. It must also stop continuing from
// the abandoned response, whose content the local context no longer matches.
func TestWSInterruptionCleansConnection(t *testing.T) {
	srv := newInterruptibleWS(t)
	svc := NewLLM(Config{APIKey: "k", WSURL: "ws" + strings.TrimPrefix(srv.URL, "http")})
	t.Cleanup(svc.disconnect)

	convo := frames.NewLLMContext("")
	convo.AddUserMessage("hello")

	// Interrupt the first turn once it has started generating.
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan error, 1)
	go func() {
		done <- svc.Generate(ctx, convo, func(string) error { return nil })
	}()
	<-srv.started
	cancel()

	if err := <-done; !errors.Is(err, context.Canceled) {
		t.Fatalf("interrupted Generate = %v, want context.Canceled", err)
	}

	srv.mu.Lock()
	canceledID := srv.canceled
	srv.mu.Unlock()
	if canceledID != "resp_1" {
		t.Errorf("canceled response = %q, want the interrupted response to be canceled server-side", canceledID)
	}

	// The connection must now be clean: the next turn reads its own output.
	convo.AddAssistantMessage("thinking")
	convo.AddUserMessage("again")

	var text strings.Builder
	if err := svc.Generate(context.Background(), convo, func(s string) error {
		text.WriteString(s)
		return nil
	}); err != nil {
		t.Fatalf("second Generate: %v", err)
	}
	if text.String() != "second" {
		t.Errorf("text = %q, want only the second turn's output", text.String())
	}

	srv.mu.Lock()
	second := srv.requests[1]
	srv.mu.Unlock()
	if _, ok := second["previous_response_id"]; ok {
		t.Error("continued from the interrupted response, whose content the context no longer matches")
	}
}
