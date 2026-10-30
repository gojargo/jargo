package llmwithbackend

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"slices"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/gojargo/jargo/adapter"
	"github.com/gojargo/jargo/classifier"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/service/llm"
	"github.com/gojargo/jargo/service/settings"
)

// streamingFrontend is a text frontend whose inference waits before its first
// output, then streams. Its output is two text chunks streaming apart, or a call
// to call with no text. It records whether an inference was canceled.
type streamingFrontend struct {
	*llm.Base
	adapter          adapter.Base
	firstOutputAfter time.Duration
	streaming        time.Duration
	call             string
	canceled         atomic.Bool
}

func newStreamingFrontend(firstOutputAfter, streaming time.Duration, call, system string) *streamingFrontend {
	f := &streamingFrontend{firstOutputAfter: firstOutputAfter, streaming: streaming, call: call}
	f.Base = llm.New("StreamingFrontend", f)
	if system == "" {
		system = "You are a voice assistant."
	}
	f.SetSystemInstruction(system)
	return f
}

func (f *streamingFrontend) LLMAdapter() llm.BuiltinToolHolder { return &f.adapter }

func (f *streamingFrontend) LLMWithBackendRoleObjection(llm.LLMWithBackendRole) string { return "" }

func (f *streamingFrontend) ServiceMetadataFrame() frames.ServiceMetadata {
	return frames.NewLLMServiceMetadataFrame(f.Name())
}

func (f *streamingFrontend) Generate(ctx context.Context, _ *frames.LLMContext, emit llm.Emit) error {
	return f.stream(ctx, emit, nil)
}

func (f *streamingFrontend) GenerateWithTools(ctx context.Context, _ *frames.LLMContext, sink llm.Sink) error {
	return f.stream(ctx, sink.Text, sink.Tool)
}

func (f *streamingFrontend) stream(
	ctx context.Context, text func(string) error, tool func(frames.ToolCall) error,
) error {
	if !f.sleep(ctx, f.firstOutputAfter) {
		return ctx.Err()
	}
	if f.call != "" && tool != nil {
		return tool(frames.ToolCall{ID: "model_call", Name: f.call, Args: json.RawMessage(`{}`)})
	}
	if err := text("Hello"); err != nil {
		return err
	}
	if !f.sleep(ctx, f.streaming) {
		return ctx.Err()
	}
	return text(" there")
}

// sleep waits for d, reporting false, and recording the cancellation, when ctx
// ends first.
func (f *streamingFrontend) sleep(ctx context.Context, d time.Duration) bool {
	select {
	case <-time.After(d):
		return true
	case <-ctx.Done():
		f.canceled.Store(true)
		return false
	}
}

// scriptedClassifier answers after a while, "yes" for the user messages in yes.
// It records each state it was asked about.
type scriptedClassifier struct {
	*classifier.Base
	after time.Duration
	yes   []string
	fail  bool

	mu     sync.Mutex
	states []classifierState
}

func newScriptedClassifier(after time.Duration, fail bool, yes ...string) *scriptedClassifier {
	c := &scriptedClassifier{after: after, yes: yes, fail: fail}
	c.Base = classifier.NewBase("ScriptedClassifier", "", c)
	return c
}

func (c *scriptedClassifier) Model() string { return "scripted" }

func (c *scriptedClassifier) AskQuestions(
	ctx context.Context, state any, questions []classifier.Named[classifier.Question],
) (map[string]classifier.Result, *frames.LLMTokenUsage, error) {
	s, _ := state.(classifierState)
	c.mu.Lock()
	c.states = append(c.states, s)
	c.mu.Unlock()
	select {
	case <-time.After(c.after):
	case <-ctx.Done():
		return nil, nil, ctx.Err()
	}
	if c.fail {
		return nil, nil, fmt.Errorf("%w: unavailable", classifier.ErrClassifier)
	}
	probability := 0.05
	if last := s.Conversation[len(s.Conversation)-1].Text; slices.Contains(c.yes, last) {
		probability = 0.95
	}
	results := map[string]classifier.Result{}
	for _, q := range questions {
		results[q.Name] = classifier.YesNoResult{Probability: probability}
	}
	return results, nil, nil
}

func (c *scriptedClassifier) asked() []classifierState {
	c.mu.Lock()
	defer c.mu.Unlock()
	return slices.Clone(c.states)
}

// recordingConnector is a connector whose delegate records each call instead of
// reaching a backend.
type recordingConnector struct {
	*BackendConnector
	mu        sync.Mutex
	delegated []string
}

func newRecordingConnector() *recordingConnector {
	r := &recordingConnector{}
	r.BackendConnector = NewBackendConnector(ConnectorConfig{
		BuildTools: func(c *BackendConnector) []frames.Tool {
			tools := c.BuildTools()
			tools[0].Handler = llm.FunctionCallHandler(func(ctx context.Context, p llm.FunctionCallParams) error {
				r.mu.Lock()
				r.delegated = append(r.delegated, p.ToolCallID)
				r.mu.Unlock()
				return p.Result(ctx, `{"status":"delegated"}`, runLLM(false))
			})
			return tools
		},
	})
	return r
}

func (r *recordingConnector) calls() []string {
	r.mu.Lock()
	defer r.mu.Unlock()
	return slices.Clone(r.delegated)
}

// earlyDelegations are the calls predictive delegation made, leaving out the
// model's own.
func (r *recordingConnector) earlyDelegations() []string {
	return slices.DeleteFunc(r.calls(), func(id string) bool { return id == "model_call" })
}

func userTurn(text string) *frames.LLMContextFrame {
	if text == "" {
		text = "Fix the flaky test."
	}
	return frames.NewLLMContextFrame(conversation(user(text)))
}

func withPredictiveDelegation(
	t *testing.T, frontend Frontend, c classifier.Classifier,
) (*LLMWithBackend, *recordingConnector) {
	t.Helper()
	connector := newRecordingConnector()
	pd, err := NewPredictiveDelegation(PredictiveDelegationConfig{Classifier: c})
	if err != nil {
		t.Fatal(err)
	}
	service, err := New(Config{
		Frontend: frontend, BackendName: "backend", Connector: connector.BackendConnector,
		PredictiveDelegation: pd,
	})
	if err != nil {
		t.Fatal(err)
	}
	return service, connector
}

func llmTexts(fs []frames.Frame) []string {
	var got []string
	for _, f := range fs {
		if tf, ok := f.(*frames.LLMTextFrame); ok {
			got = append(got, tf.Text)
		}
	}
	return got
}

const ms = time.Millisecond

func TestAYesBeforeTheFrontendsFirstOutputDelegatesTheTurnEarly(t *testing.T) {
	frontend := newStreamingFrontend(400*ms, 0, "", "")
	service, connector := withPredictiveDelegation(t, frontend,
		newScriptedClassifier(50*ms, false, "Fix the flaky test."))

	down := runSteps(t, service,
		userTurn(""), frames.NewTextFrame("queued behind the inference"), 600*ms)

	if got := connector.earlyDelegations(); len(got) != 1 {
		t.Errorf("early delegations = %v, want one", got)
	}
	if !frontend.canceled.Load() {
		t.Error("the frontend's inference was not canceled")
	}
	if got := llmTexts(down); len(got) != 0 {
		t.Errorf("texts = %v, want none", got)
	}
	// What was queued behind the canceled inference is still processed.
	var queued []string
	for _, f := range down {
		if tf, ok := f.(*frames.TextFrame); ok {
			queued = append(queued, tf.Text)
		}
	}
	if !slices.Equal(queued, []string{"queued behind the inference"}) {
		t.Errorf("text frames = %v", queued)
	}
	// Downstream, the call belongs to the canceled inference's response, as a
	// model's own call would: announced inside it, and run from then on, so its
	// in-progress report may come after the end of the response.
	var shape []string
	for _, f := range down {
		switch f.(type) {
		case *frames.LLMFullResponseStartFrame:
			shape = append(shape, "start")
		case *frames.FunctionCallsStartedFrame:
			shape = append(shape, "started")
		case *frames.FunctionCallInProgressFrame:
			shape = append(shape, "in progress")
		case *frames.LLMFullResponseEndFrame:
			shape = append(shape, "end")
		}
	}
	if len(shape) != 4 || shape[0] != "start" || shape[1] != "started" ||
		!slices.Contains(shape[2:], "in progress") || !slices.Contains(shape[2:], "end") {
		t.Errorf("shape = %v, want start, started, then the end and the call in progress", shape)
	}
}

func TestAYesAfterTheFrontendsFirstTextLeavesTheTurnToTheFrontend(t *testing.T) {
	frontend := newStreamingFrontend(50*ms, 300*ms, "", "")
	service, connector := withPredictiveDelegation(t, frontend,
		newScriptedClassifier(150*ms, false, "Fix the flaky test."))

	down := runSteps(t, service, userTurn(""), 500*ms)

	if got := connector.calls(); len(got) != 0 {
		t.Errorf("delegated %v, want nothing", got)
	}
	if got := llmTexts(down); !slices.Equal(got, []string{"Hello", " there"}) {
		t.Errorf("texts = %v", got)
	}
}

func TestAYesAfterTheFrontendsOwnDelegateCallAddsNothing(t *testing.T) {
	frontend := newStreamingFrontend(50*ms, 0, DelegateToolName, "")
	service, connector := withPredictiveDelegation(t, frontend,
		newScriptedClassifier(150*ms, false, "Fix the flaky test."))

	runSteps(t, service, userTurn(""), 300*ms)

	if got := connector.calls(); !slices.Equal(got, []string{"model_call"}) {
		t.Errorf("delegated %v, want the model's own call only", got)
	}
}

func TestANoLeavesTheTurnToTheFrontend(t *testing.T) {
	frontend := newStreamingFrontend(200*ms, 0, "", "")
	service, connector := withPredictiveDelegation(t, frontend, newScriptedClassifier(50*ms, false))

	down := runSteps(t, service, userTurn(""), 300*ms)

	if got := connector.calls(); len(got) != 0 {
		t.Errorf("delegated %v, want nothing", got)
	}
	if got := llmTexts(down); !slices.Equal(got, []string{"Hello", " there"}) {
		t.Errorf("texts = %v", got)
	}
}

func TestAFailingClassifierLeavesTheTurnToTheFrontend(t *testing.T) {
	frontend := newStreamingFrontend(200*ms, 0, "", "")
	service, connector := withPredictiveDelegation(t, frontend, newScriptedClassifier(50*ms, true))

	down := runSteps(t, service, userTurn(""), 300*ms)

	if got := connector.calls(); len(got) != 0 {
		t.Errorf("delegated %v, want nothing", got)
	}
	if got := llmTexts(down); !slices.Equal(got, []string{"Hello", " there"}) {
		t.Errorf("texts = %v", got)
	}
}

func TestAYesForATurnStillQueuedIsCarriedOutOnceItStarts(t *testing.T) {
	frontend := newStreamingFrontend(300*ms, 0, "", "")
	service, connector := withPredictiveDelegation(t, frontend,
		newScriptedClassifier(50*ms, false, "Fix the flaky test."))

	down := runSteps(t, service, userTurn("Tell me a joke."), userTurn("Fix the flaky test."), 800*ms)

	// The earlier response ran to the end; the queued turn was delegated early
	// once its inference started.
	if got := llmTexts(down); !slices.Equal(got, []string{"Hello", " there"}) {
		t.Errorf("texts = %v", got)
	}
	if got := connector.earlyDelegations(); len(got) != 1 {
		t.Errorf("early delegations = %v, want one", got)
	}
	if !frontend.canceled.Load() {
		t.Error("the frontend's inference was not canceled")
	}
}

func TestAnInterruptionLeavesTheTurnToTheFrontend(t *testing.T) {
	frontend := newStreamingFrontend(300*ms, 0, "", "")
	service, connector := withPredictiveDelegation(t, frontend,
		newScriptedClassifier(200*ms, false, "Fix the flaky test."))

	runSteps(t, service, userTurn(""), 100*ms, frames.NewInterruptionFrame(), 400*ms)

	if got := connector.calls(); len(got) != 0 {
		t.Errorf("delegated %v, want nothing", got)
	}
}

func TestOnlyANewUserMessageIsRaced(t *testing.T) {
	frontend := newStreamingFrontend(0, 0, "", "")
	c := newScriptedClassifier(0, false)
	service, _ := withPredictiveDelegation(t, frontend, c)
	turn := userTurn("")
	developer := frames.NewLLMContextFrame(conversation(frames.Message{
		Role: frames.RoleDeveloper, Text: "Greet the user.",
	}))
	speculative := frames.NewLLMContextFrame(conversation(user("Fix the")))
	speculative.Speculation = true

	runSteps(t, service, developer, speculative, turn, 100*ms, turn, 100*ms)

	// The developer message and the speculative turn are not raced, and a turn
	// raced once is not raced again.
	if got := c.asked(); len(got) != 1 {
		t.Errorf("the classifier was asked %d times, want once", len(got))
	}
}

func TestNoTurnIsRacedWhileTheFrontendFiltersIncompleteTurns(t *testing.T) {
	frontend := newStreamingFrontend(0, 0, "", "")
	c := newScriptedClassifier(0, false)
	service, _ := withPredictiveDelegation(t, frontend, c)

	runSteps(t, service,
		frames.NewLLMUpdateSettingsFrame(&settings.LLM{FilterIncompleteUserTurns: settings.Set(true)}),
		100*ms, userTurn(""), 100*ms)

	if got := c.asked(); len(got) != 0 {
		t.Errorf("the classifier was asked about %v, want nothing", got)
	}
}

func TestTheClassifierSeesTheFrontendsInstructionAndTheLatestConversation(t *testing.T) {
	frontend := newStreamingFrontend(0, 0, "", "You are Pip.")
	c := newScriptedClassifier(0, false)
	service, _ := withPredictiveDelegation(t, frontend, c)
	convo := conversation(user("Fix the flaky test."))
	convo.AddAssistantToolCall(frames.ToolCall{ID: "c1", Name: DelegateToolName, Args: json.RawMessage(`{}`)})
	convo.AddToolResult(frames.ToolResult{ID: "c1", Name: DelegateToolName, Content: `{"status": "delegated"}`})
	convo.AddMessage(frames.Message{
		Role: frames.RoleDeveloper, Text: "Backend: Fixed it.\n\n" + BackendMessageRelayInstruction,
	})
	convo.AddMessage(user("Thanks!"))

	runSteps(t, service, frames.NewLLMContextFrame(convo), 100*ms)

	states := c.asked()
	if len(states) != 1 {
		t.Fatalf("the classifier was asked %d times, want once", len(states))
	}
	// What the frontend's model reads: the app's instruction, then the pairing
	// instruction appended to it.
	if !strings.HasPrefix(states[0].Assistant, "You are Pip.\n\nTHE BACKEND:") {
		t.Errorf("assistant = %q", states[0].Assistant[:min(60, len(states[0].Assistant))])
	}
	got, err := json.Marshal(states[0].Conversation)
	if err != nil {
		t.Fatal(err)
	}
	want := `[{"speaker":"user","text":"Fix the flaky test."},` +
		`{"speaker":"assistant","action":"handed the request to the backend"},` +
		`{"speaker":"backend","text":"Backend: Fixed it."},` +
		`{"speaker":"user","text":"Thanks!"}]`
	if string(got) != want {
		t.Errorf("conversation = %s\nwant %s", got, want)
	}
}

func TestTheClassifiersMetricsReachThePipeline(t *testing.T) {
	frontend := newStreamingFrontend(0, 0, "", "")
	service, _ := withPredictiveDelegation(t, frontend, newScriptedClassifier(0, false))

	down := runSteps(t, service, userTurn(""), 100*ms)

	if !slices.ContainsFunc(down, func(f frames.Frame) bool {
		_, ok := f.(*frames.MetricsFrame)
		return ok
	}) {
		t.Error("no metrics frame reached the end of the pipeline")
	}
}

func TestAnEarlyDelegationIsSettledLikeTheModelsOwn(t *testing.T) {
	for _, respond := range []bool{true, false} {
		t.Run(fmt.Sprint("respond_on_delegate=", respond), func(t *testing.T) {
			backend := weatherBackend(t, scriptedRun{text: ">> On it."})
			pd, err := NewPredictiveDelegation(PredictiveDelegationConfig{
				Classifier: newScriptedClassifier(50*ms, false, "Fix the flaky test."),
			})
			if err != nil {
				t.Fatal(err)
			}
			service, err := New(Config{
				Frontend:             newStreamingFrontend(time.Second, 0, "", ""),
				Backend:              backend,
				Connector:            NewBackendConnector(ConnectorConfig{RespondOnDelegate: &respond}),
				PredictiveDelegation: pd,
			})
			if err != nil {
				t.Fatal(err)
			}

			down := runSteps(t, service, userTurn(""), 1500*ms)

			var results []*frames.FunctionCallResultFrame
			for _, f := range down {
				if r, ok := f.(*frames.FunctionCallResultFrame); ok {
					results = append(results, r)
				}
			}
			if len(results) != 1 {
				t.Fatalf("results = %+v, want one", results)
			}
			var result struct {
				Status string `json:"status"`
			}
			if err := json.Unmarshal([]byte(results[0].Result), &result); err != nil || result.Status != "delegated" {
				t.Errorf("result = %s", results[0].Result)
			}
			props := results[0].Properties
			if props == nil || props.RunLLM == nil || *props.RunLLM != respond {
				t.Errorf("properties = %+v, want RunLLM %v", props, respond)
			}
		})
	}
}

func TestPredictiveDelegationNeedsATextFrontend(t *testing.T) {
	frontend := newTextFrontend()
	frontend.realtime = true
	pd, err := NewPredictiveDelegation(PredictiveDelegationConfig{Classifier: newScriptedClassifier(0, false)})
	if err != nil {
		t.Fatal(err)
	}
	_, err = New(Config{
		Frontend: frontend, BackendName: "backend",
		Connector: NewBackendConnector(ConnectorConfig{
			RequestStrategy: NewExplicitBackendRequestStrategy(StrategyTexts{}),
		}),
		PredictiveDelegation: pd,
	})
	if !errors.Is(err, ErrPredictiveDelegationNeedsTextFrontend) || !strings.Contains(err.Error(), "text frontend") {
		t.Errorf("err = %v", err)
	}
}

func TestPredictiveDelegationNeedsADelegateCallWithNoArguments(t *testing.T) {
	pd, err := NewPredictiveDelegation(PredictiveDelegationConfig{Classifier: newScriptedClassifier(0, false)})
	if err != nil {
		t.Fatal(err)
	}
	_, err = New(Config{
		Frontend: newTextFrontend(), BackendName: "backend",
		Connector: NewBackendConnector(ConnectorConfig{
			RequestStrategy: NewExplicitBackendRequestStrategy(StrategyTexts{}),
		}),
		PredictiveDelegation: pd,
	})
	if !errors.Is(err, ErrPredictiveDelegationNeedsNoArguments) ||
		!strings.Contains(err.Error(), "ExplicitBackendRequestStrategy") {
		t.Errorf("err = %v", err)
	}
}
