package llmwithbackend

import (
	"context"
	"encoding/json"
	"errors"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/gojargo/jargo/adapter"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/processor/rtvi"
	"github.com/gojargo/jargo/service/llm"
	"github.com/gojargo/jargo/utils/events"
	"github.com/gojargo/jargo/workers"
	"github.com/gojargo/jargo/workers/llmworker"
)

// Ported from upstream's LLMWithBackend suite. The connector is exercised
// directly with a faked session. The service is exercised with a scripted
// backend under a real runner, so the whole path runs: the tool installed on the
// frontend, delegate called, the message sent over the session, and the
// backend's outputs appended to the frontend's conversation.

// textFrontend is a text frontend that forwards every frame, and converts
// through an adapter of its own, which is where its built-in tools go.
type textFrontend struct {
	*llm.Base
	adapter  adapter.Base
	realtime bool
	// delegateOnce calls delegate on the first context frame, with these
	// arguments.
	delegateOnce bool
	arguments    string
	delegated    bool
	// onStart is called when the frontend gets its StartFrame.
	onStart func()
	// objection is what the frontend says about a role, if anything.
	objection string
}

func newTextFrontend() *textFrontend {
	f := &textFrontend{}
	f.Base = llm.New("TextFrontend", f)
	f.SetSystemInstruction("You are a voice assistant.")
	return f
}

func newDelegatingFrontend(arguments string) *textFrontend {
	f := newTextFrontend()
	f.delegateOnce, f.arguments = true, arguments
	return f
}

func (f *textFrontend) Generate(context.Context, *frames.LLMContext, llm.Emit) error { return nil }

func (f *textFrontend) LLMAdapter() llm.BuiltinToolHolder { return &f.adapter }

func (f *textFrontend) LLMWithBackendRoleObjection(role llm.LLMWithBackendRole) string {
	if f.objection == "" {
		return ""
	}
	return "cannot be the " + string(role) + ": " + f.objection
}

func (f *textFrontend) ServiceMetadataFrame() frames.ServiceMetadata {
	md := frames.NewLLMServiceMetadataFrame(f.Name())
	md.Realtime = f.realtime
	return md
}

func (f *textFrontend) ProcessFrame(ctx context.Context, fr frames.Frame, dir processor.Direction) error {
	if _, ok := fr.(*frames.StartFrame); ok && f.onStart != nil {
		f.onStart()
	}
	cf, isContext := fr.(*frames.LLMContextFrame)
	if !isContext || !f.delegateOnce || f.delegated {
		if err := f.Base.Base.ProcessFrame(ctx, fr, dir); err != nil {
			return err
		}
		return f.PushFrame(ctx, fr, dir)
	}
	if err := f.Base.Base.ProcessFrame(ctx, fr, dir); err != nil {
		return err
	}
	f.delegated = true
	args := f.arguments
	if args == "" {
		args = `{}`
	}
	if err := f.PushFrame(ctx, frames.NewLLMFullResponseStartFrame(), processor.Downstream); err != nil {
		return err
	}
	if err := f.RunFunctionCalls(ctx, cf.Context, []frames.ToolCall{
		{ID: "call_1", Name: DelegateToolName, Args: json.RawMessage(args)},
	}); err != nil {
		return err
	}
	return f.PushFrame(ctx, frames.NewLLMFullResponseEndFrame(), processor.Downstream)
}

// fakeSession stands in for the session with the backend.
type fakeSession struct {
	mu       sync.Mutex
	requests []string
	status   string
}

func (s *fakeSession) Send(_ context.Context, request string) (string, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.requests = append(s.requests, request)
	if s.status == "" {
		return "idle", nil
	}
	return s.status, nil
}

func bound(t *testing.T, c *BackendConnector, realtime bool, session *fakeSession) *BackendConnector {
	t.Helper()
	if c == nil {
		c = NewBackendConnector(ConnectorConfig{})
	}
	if err := c.Bind(ConnectorContext{BackendName: "backend", FrontendIsRealtime: realtime}); err != nil {
		t.Fatal(err)
	}
	if session != nil {
		c.setSession(session)
	}
	return c
}

// settled is one result a call was settled with.
type settled struct {
	result string
	runLLM *bool
}

// params builds a delegate call over convo, recording what it is settled with.
func params(convo *frames.LLMContext, arguments string) (llm.FunctionCallParams, *[]settled) {
	if convo == nil {
		convo = frames.NewLLMContext("")
	}
	if arguments == "" {
		arguments = `{}`
	}
	var got []settled
	return llm.FunctionCallParams{
		FunctionName: DelegateToolName,
		ToolCallID:   "call_1",
		Arguments:    json.RawMessage(arguments),
		Context:      convo,
		Result: func(_ context.Context, result string, props *frames.FunctionCallResultProperties) error {
			s := settled{result: result}
			if props != nil {
				s.runLLM = props.RunLLM
			}
			got = append(got, s)
			return nil
		},
	}, &got
}

func conversation(messages ...frames.Message) *frames.LLMContext {
	convo := frames.NewLLMContext("")
	for _, m := range messages {
		convo.AddMessage(m)
	}
	return convo
}

func user(text string) frames.Message { return frames.Message{Role: frames.RoleUser, Text: text} }
func assistant(text string) frames.Message {
	return frames.Message{Role: frames.RoleAssistant, Text: text}
}

// --- Connector defaults and the tools it builds ---

func TestATextFrontendHandsOverTheTranscript(t *testing.T) {
	c := bound(t, nil, false, nil)
	if _, ok := c.RequestStrategy().(*TranscriptBackendRequestStrategy); !ok {
		t.Errorf("strategy = %T, want the transcript one", c.RequestStrategy())
	}
	var names []string
	for _, tool := range c.Tools() {
		names = append(names, tool.Name)
		if tool.Handler == nil {
			t.Errorf("%s carries no handler", tool.Name)
		}
	}
	if !slices.Equal(names, []string{"delegate"}) {
		t.Errorf("tools = %v", names)
	}
	var schema struct {
		Properties map[string]any `json:"properties"`
	}
	if err := json.Unmarshal(c.Tool().Parameters, &schema); err != nil {
		t.Fatal(err)
	}
	if len(schema.Properties) != 0 {
		t.Errorf("properties = %v, want none", schema.Properties)
	}
}

func TestARealtimeFrontendWordsTheRequest(t *testing.T) {
	c := bound(t, nil, true, nil)
	if _, ok := c.RequestStrategy().(*ExplicitBackendRequestStrategy); !ok {
		t.Errorf("strategy = %T, want the explicit one", c.RequestStrategy())
	}
	var schema struct {
		Required []string `json:"required"`
	}
	if err := json.Unmarshal(c.Tool().Parameters, &schema); err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(schema.Required, []string{"request"}) {
		t.Errorf("required = %v", schema.Required)
	}
}

func TestARealtimeFrontendRefusesTheTranscriptStrategy(t *testing.T) {
	c := NewBackendConnector(ConnectorConfig{
		RequestStrategy: NewTranscriptBackendRequestStrategy(StrategyTexts{}),
	})
	err := c.Bind(ConnectorContext{BackendName: "backend", FrontendIsRealtime: true})
	if err == nil || !strings.Contains(err.Error(), "TranscriptBackendRequestStrategy") {
		t.Errorf("err = %v, want the transcript strategy named", err)
	}
}

func TestARealtimeFrontendTakesAGivenExplicitStrategy(t *testing.T) {
	strategy := NewExplicitBackendRequestStrategy(StrategyTexts{})
	c := bound(t, NewBackendConnector(ConnectorConfig{RequestStrategy: strategy}), true, nil)
	if c.RequestStrategy() != strategy {
		t.Error("the given strategy was replaced")
	}
}

func TestTheToolDescriptionFramesAHandoff(t *testing.T) {
	if c := bound(t, nil, false, nil); !strings.Contains(c.Tool().Description, "One handoff per reply") {
		t.Errorf("description = %q", c.Tool().Description)
	}
}

func TestTheFrontendGuidanceCoversDelegationAndTheBackendsMessages(t *testing.T) {
	guidance := bound(t, nil, false, nil).FrontendInstruction()
	for _, want := range []string{
		"delegate tool", `marked "Backend:"`, `"Backend (working):"`, "to start, change or stop its work",
	} {
		if !strings.Contains(guidance, want) {
			t.Errorf("guidance lacks %q", want)
		}
	}
}

func TestTheStrategysGuidanceSitsInsideTheDelegatingParagraph(t *testing.T) {
	guidance := bound(t, nil, false, nil).FrontendInstruction()
	if !strings.HasPrefix(guidance, "THE BACKEND:") {
		t.Errorf("guidance starts %q", guidance[:20])
	}
	delegating, _, _ := strings.Cut(guidance, "\n\n")
	if !strings.HasSuffix(delegating, " "+TranscriptFrontendInstruction) {
		t.Errorf("the delegating paragraph ends %q", delegating[len(delegating)-60:])
	}
	if strings.Contains(guidance, PairingInstructionStrategyPlaceholder) {
		t.Error("the placeholder was left in")
	}
	one, after, messages := strings.Index(guidance, "ONE ASSISTANT:"),
		strings.Index(guidance, "AFTER DELEGATING:"), strings.Index(guidance, "BACKEND MESSAGES:")
	if one >= after || after >= messages {
		t.Errorf("paragraphs out of order: %d %d %d", one, after, messages)
	}
	// The instruction ends by restating its rule for messages about delegated
	// work.
	paragraphs := strings.Split(guidance, "\n\n")
	if last := paragraphs[len(paragraphs)-1]; !strings.HasPrefix(last,
		"Whatever else a reply does, a user message about work") {
		t.Errorf("the last paragraph is %q", last)
	}
}

func TestAnAppCanReplaceTheFrontendsPairingInstruction(t *testing.T) {
	placed := bound(t, NewBackendConnector(ConnectorConfig{
		FrontendPairingInstruction: "Hand over: {strategy} Then talk.",
	}), false, nil)
	if got, want := placed.FrontendInstruction(), "Hand over: "+TranscriptFrontendInstruction+" Then talk."; got != want {
		t.Errorf("placed = %q, want %q", got, want)
	}
	appended := bound(t, NewBackendConnector(ConnectorConfig{
		FrontendPairingInstruction: "Hand things over.",
	}), false, nil)
	if got, want := appended.FrontendInstruction(), "Hand things over.\n\n"+TranscriptFrontendInstruction; got != want {
		t.Errorf("appended = %q, want %q", got, want)
	}
}

func TestAStrategyTakesItsThreeTextsAsArguments(t *testing.T) {
	strategy := NewExplicitBackendRequestStrategy(StrategyTexts{
		FrontendInstruction: "Say it plainly.",
		ToolDescription:     "Hand it over.",
		RequestInstruction:  "Do it now.",
	})
	c := bound(t, NewBackendConnector(ConnectorConfig{RequestStrategy: strategy}), false, nil)
	if !strings.Contains(c.FrontendInstruction(), "waiting. Say it plainly.") {
		t.Error("the frontend instruction was not placed")
	}
	if c.Tool().Description != "Hand it over." {
		t.Errorf("description = %q", c.Tool().Description)
	}
	p, _ := params(nil, `{"request":"Book a taxi"}`)
	if got, _ := strategy.ComposeRequest(t.Context(), p); got != "Book a taxi\n\nDo it now." {
		t.Errorf("request = %q", got)
	}
}

// --- Request strategies ---

// TestTheTranscriptRequestSkipsMessagesInAServicesOwnFormat checks a frontend
// that records its calls in its own format has them left out.
func TestTheTranscriptRequestSkipsMessagesInAServicesOwnFormat(t *testing.T) {
	session := &fakeSession{}
	c := bound(t, nil, false, session)
	convo := conversation(
		user("weather in seattle?"),
		frames.NewLLMSpecificMessage("google", map[string]any{"parts": []any{map[string]any{"text": "call"}}}),
		assistant("Let me check."),
	)

	p, _ := params(convo, "")
	if err := c.Delegate(t.Context(), p); err != nil {
		t.Fatal(err)
	}

	want := "Voice conversation so far:\n" +
		"USER: weather in seattle?\n" +
		"ASSISTANT: Let me check.\n" +
		"\n" + llmworker.TranscriptRequestInstruction
	if !slices.Equal(session.requests, []string{want}) {
		t.Errorf("requests = %q", session.requests)
	}
}

func TestTheTranscriptRequestSendsOnlyWhatTheBackendHasNotSeen(t *testing.T) {
	session := &fakeSession{}
	c := bound(t, nil, false, session)
	convo := conversation(user("weather in seattle?"))

	p, _ := params(convo, "")
	if err := c.Delegate(t.Context(), p); err != nil {
		t.Fatal(err)
	}
	convo.AddMessage(assistant("It's raining."))
	convo.AddMessage(frames.Message{Role: frames.RoleDeveloper, Text: "Backend: Looking it up."})
	convo.AddMessage(user("and boston?"))
	p, _ = params(convo, "")
	if err := c.Delegate(t.Context(), p); err != nil {
		t.Fatal(err)
	}

	if !strings.HasPrefix(session.requests[0], "Voice conversation so far:\nUSER: weather in seattle?\n") {
		t.Errorf("first request = %q", session.requests[0])
	}
	// The backend's own messages are not sent back to it.
	want := "Voice conversation since the previous delegation:\n" +
		"ASSISTANT: It's raining.\n" +
		"USER: and boston?\n" +
		"\n" + llmworker.TranscriptRequestInstruction
	if session.requests[1] != want {
		t.Errorf("second request = %q", session.requests[1])
	}
}

// TestADelegateCallWithNothingNewSendsNothing checks a second call for the same
// turn finds the transcript slice empty and settles quietly.
func TestADelegateCallWithNothingNewSendsNothing(t *testing.T) {
	session := &fakeSession{}
	c := bound(t, nil, false, session)
	convo := conversation(user("weather in seattle?"))

	p, _ := params(convo, "")
	if err := c.Delegate(t.Context(), p); err != nil {
		t.Fatal(err)
	}
	p, got := params(convo, "")
	if err := c.Delegate(t.Context(), p); err != nil {
		t.Fatal(err)
	}

	if len(session.requests) != 1 {
		t.Errorf("requests = %q, want the first only", session.requests)
	}
	if len(*got) != 1 || (*got)[0].result != `{"status":"already_delegated"}` ||
		(*got)[0].runLLM == nil || *(*got)[0].runLLM {
		t.Errorf("settled with %+v", *got)
	}
}

func TestTheExplicitRequestSendsTheModelsWords(t *testing.T) {
	session := &fakeSession{}
	c := bound(t, NewBackendConnector(ConnectorConfig{
		RequestStrategy: NewExplicitBackendRequestStrategy(StrategyTexts{}),
	}), false, session)

	p, _ := params(nil, `{"request":"Weather in Seattle, Fahrenheit."}`)
	if err := c.Delegate(t.Context(), p); err != nil {
		t.Fatal(err)
	}

	want := "Weather in Seattle, Fahrenheit.\n\n" + llmworker.ExplicitRequestInstruction
	if !slices.Equal(session.requests, []string{want}) {
		t.Errorf("requests = %q", session.requests)
	}
}

// --- The delegate tool ---

func TestDelegateSettlesAtOnceWithWhatTheBackendWasDoing(t *testing.T) {
	for _, status := range []string{"idle", "working"} {
		p, got := params(conversation(user("do it")), "")
		if err := bound(t, nil, false, &fakeSession{status: status}).Delegate(t.Context(), p); err != nil {
			t.Fatal(err)
		}
		want := `{"backend":"` + status + `","status":"delegated"}`
		if len(*got) != 1 || (*got)[0].result != want || (*got)[0].runLLM == nil || !*(*got)[0].runLLM {
			t.Errorf("%s: settled with %+v", status, *got)
		}
	}
}

func TestDelegateCanLeaveTheFrontendQuiet(t *testing.T) {
	quiet := false
	p, got := params(conversation(user("do it")), "")
	c := bound(t, NewBackendConnector(ConnectorConfig{RespondOnDelegate: &quiet}), false, &fakeSession{})
	if err := c.Delegate(t.Context(), p); err != nil {
		t.Fatal(err)
	}
	if len(*got) != 1 || (*got)[0].runLLM == nil || *(*got)[0].runLLM {
		t.Errorf("settled with %+v, want no run", *got)
	}
}

func TestDelegateFailsWhenTheBackendNeverAttaches(t *testing.T) {
	c := bound(t, NewBackendConnector(ConnectorConfig{Timeout: 50 * time.Millisecond}), false, nil)
	p, got := params(conversation(user("do it")), "")

	if err := c.Delegate(t.Context(), p); !errors.Is(err, errNotAttached) {
		t.Errorf("err = %v, want not attached", err)
	}
	if len(*got) != 0 {
		t.Errorf("settled with %+v, want nothing", *got)
	}
}

// --- Delivery ---

// recordingFrontend records what the connector queues into it and pushes from
// it.
type recordingFrontend struct {
	mu     sync.Mutex
	queued []frames.Frame
	pushed []frames.Frame
}

func (r *recordingFrontend) QueueFrame(_ context.Context, f frames.Frame, _ processor.Direction) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.queued = append(r.queued, f)
	return nil
}

func (r *recordingFrontend) PushFrame(_ context.Context, f frames.Frame, _ processor.Direction) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.pushed = append(r.pushed, f)
	return nil
}

// appended is the messages queued, each with whether it runs the model.
func (r *recordingFrontend) appended() []appendedMessage {
	r.mu.Lock()
	defer r.mu.Unlock()
	var out []appendedMessage
	for _, f := range r.queued {
		if af, ok := f.(*frames.LLMMessagesAppendFrame); ok {
			out = append(out, appendedMessage{
				role: af.Messages[0].Role, text: af.Messages[0].Text, run: af.RunLLM,
				interruptible: frames.Interruptible(af),
			})
		}
	}
	return out
}

type appendedMessage struct {
	role          frames.Role
	text          string
	run           bool
	interruptible bool
}

func TestOutputsAreAppendedToTheFrontendAndRunItAsFlagged(t *testing.T) {
	c := bound(t, nil, false, nil)
	frontend := &recordingFrontend{}

	c.Deliver(t.Context(), frontend, llmworker.BackendOutput{Text: "Let me check."})
	c.Deliver(t.Context(), frontend, llmworker.BackendOutput{Text: "Weather first.", IsThought: true})
	c.Deliver(t.Context(), frontend, llmworker.NewBackendOutput("It's 62 and raining."))
	c.Deliver(t.Context(), frontend, llmworker.NewBackendOutput(""))

	want := []appendedMessage{
		{frames.RoleDeveloper, "Backend (note): Let me check.", false, false},
		{frames.RoleDeveloper, "Backend (thinking): Weather first.", false, false},
		{frames.RoleDeveloper, "Backend: It's 62 and raining.\n\n" + BackendMessageRelayInstruction, true, false},
	}
	if got := frontend.appended(); !slices.Equal(got, want) {
		t.Errorf("appended %+v, want %+v", got, want)
	}
	if len(frontend.pushed) != 0 {
		t.Errorf("pushed %v, want nothing", frontend.pushed)
	}
}

func TestTheBackendsCallsAreReportedAndNothingElseHappensToThem(t *testing.T) {
	c := bound(t, nil, false, nil)
	frontend := &recordingFrontend{}

	c.Deliver(t.Context(), frontend, llmworker.BackendToolCall{
		Phase: llmworker.BackendToolCallInProgress, FunctionName: "get_weather", ToolCallID: "toolu_1",
		Arguments: json.RawMessage(`{"location":"Seattle"}`),
	})

	if len(frontend.pushed) != 1 {
		t.Fatalf("pushed %v, want one report", frontend.pushed)
	}
	pushed, ok := frontend.pushed[0].(*frames.ExternalFunctionCallInProgressFrame)
	if !ok || pushed.ToolName != "get_weather" || pushed.ToolCallID != "toolu_1" {
		t.Errorf("pushed %v", frontend.pushed[0])
	}
	// The call is also recorded in the conversation, silently, so the frontend
	// can say what the backend is doing.
	want := []appendedMessage{
		{frames.RoleDeveloper, "Backend (working): get_weather(location='Seattle')", false, false},
	}
	if got := frontend.appended(); !slices.Equal(got, want) {
		t.Errorf("appended %+v, want %+v", got, want)
	}

	frontend = &recordingFrontend{}
	c.Deliver(t.Context(), frontend, llmworker.BackendToolCall{
		Phase: llmworker.BackendToolCallResult, FunctionName: "get_weather", ToolCallID: "toolu_1", Result: "{}",
	})
	if len(frontend.queued) != 0 {
		t.Errorf("queued %v, want nothing", frontend.queued)
	}
}

func TestABackendErrorIsSpokenAndIdleIsNot(t *testing.T) {
	c := bound(t, nil, false, nil)
	frontend := &recordingFrontend{}

	c.Deliver(t.Context(), frontend, llmworker.BackendIdle{})
	c.Deliver(t.Context(), frontend, llmworker.BackendError{Error: "provider down"})

	got := frontend.appended()
	if len(got) != 1 || !got[0].run || !strings.HasPrefix(got[0].text, "Backend: The work could not be completed") {
		t.Errorf("appended %+v", got)
	}
}

func TestRenderOutputIsTheSeamForAnotherWording(t *testing.T) {
	terse := func(o llmworker.BackendOutput) (frames.Message, bool) {
		if o.IsThought {
			return frames.Message{}, false
		}
		return frames.Message{Role: frames.RoleUser, Text: "[be] " + o.Text}, true
	}
	c := bound(t, NewBackendConnector(ConnectorConfig{RenderOutput: terse}), false, nil)
	frontend := &recordingFrontend{}

	c.Deliver(t.Context(), frontend, llmworker.BackendOutput{Text: "thinking", IsThought: true})
	c.Deliver(t.Context(), frontend, llmworker.NewBackendOutput("done"))

	want := []appendedMessage{{frames.RoleUser, "[be] done", true, false}}
	if got := frontend.appended(); !slices.Equal(got, want) {
		t.Errorf("appended %+v, want %+v", got, want)
	}
}

// --- The service ---

func TestAFrontendThatDeclinesTheRoleIsRefused(t *testing.T) {
	frontend := newTextFrontend()
	frontend.objection = "it has a backend of its own"
	_, err := New(Config{Frontend: frontend, BackendName: "backend"})
	if !errors.Is(err, ErrFrontendDeclined) || !strings.Contains(err.Error(), "cannot be the frontend") {
		t.Errorf("err = %v", err)
	}
}

func TestTheServiceAddsTheConnectorGuidanceToTheFrontendPrompt(t *testing.T) {
	frontend := newTextFrontend()
	if _, err := New(Config{Frontend: frontend, BackendName: "backend"}); err != nil {
		t.Fatal(err)
	}
	composed := frontend.SystemInstruction()
	if !strings.HasPrefix(composed, "You are a voice assistant.") || !strings.Contains(composed, "delegate tool") {
		t.Errorf("composed %q", composed)
	}
}

func toolNames(tools []frames.Tool) []string {
	names := make([]string, 0, len(tools))
	for _, t := range tools {
		names = append(names, t.Name)
	}
	return names
}

func TestTheToolsAreBuiltInBesideTheFrontendsOwn(t *testing.T) {
	frontend := newTextFrontend()
	if _, err := New(Config{Frontend: frontend, BackendName: "backend"}); err != nil {
		t.Fatal(err)
	}
	convo := frames.NewLLMContext("")
	convo.SetTools([]frames.Tool{{Name: "get_current_time", Parameters: json.RawMessage(`{"type":"object"}`)}})

	// Never in the context's tool set, so no diff sees it.
	if got := toolNames(convo.Tools()); !slices.Equal(got, []string{"get_current_time"}) {
		t.Errorf("context tools = %v", got)
	}
	// Sent on every inference all the same, beside whatever tools there are.
	if got := toolNames(frontend.adapter.WithBuiltins(convo.ToolsSchema()).Standard); !slices.Equal(got,
		[]string{"get_current_time", "delegate"}) {
		t.Errorf("sent with the context's tools: %v", got)
	}
	if got := toolNames(frontend.adapter.WithBuiltins(frames.ToolsSchema{}).Standard); !slices.Equal(got,
		[]string{"delegate"}) {
		t.Errorf("sent alone: %v", got)
	}
	if !frontend.HasFunction("delegate") {
		t.Error("delegate has no handler")
	}
}

// scriptedLLM plays one scripted response per context frame: text, and then
// function calls.
type scriptedLLM struct {
	*llm.Base

	mu   sync.Mutex
	runs []scriptedRun
}

type scriptedRun struct {
	text  string
	calls []frames.ToolCall
}

func newScriptedLLM(runs ...scriptedRun) *scriptedLLM {
	s := &scriptedLLM{runs: runs}
	s.Base = llm.New("ScriptedLLM", s)
	return s
}

func (s *scriptedLLM) Generate(context.Context, *frames.LLMContext, llm.Emit) error { return nil }

func (s *scriptedLLM) ProcessFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	fr, ok := f.(*frames.LLMContextFrame)
	if !ok {
		return s.Base.ProcessFrame(ctx, f, dir)
	}
	if err := s.Base.Base.ProcessFrame(ctx, f, dir); err != nil {
		return err
	}
	s.SyncToolHandlers(ctx, fr.Context.Tools())
	s.mu.Lock()
	var run scriptedRun
	if len(s.runs) > 0 {
		run, s.runs = s.runs[0], s.runs[1:]
	}
	s.mu.Unlock()
	if err := s.PushFrame(ctx, frames.NewLLMFullResponseStartFrame(), processor.Downstream); err != nil {
		return err
	}
	if run.text != "" {
		if err := s.PushFrame(ctx, frames.NewLLMTextFrame(run.text), processor.Downstream); err != nil {
			return err
		}
	}
	end := func() error {
		return s.PushFrame(ctx, frames.NewLLMFullResponseEndFrame(), processor.Downstream)
	}
	if len(run.calls) == 0 {
		return end()
	}
	// As the base does: announced before the response ends, run once it has.
	if err := end(); err != nil {
		return err
	}
	return s.RunFunctionCalls(ctx, fr.Context, run.calls)
}

func weatherBackend(t *testing.T, runs ...scriptedRun) *llmworker.BackendWorker {
	t.Helper()
	convo := frames.NewLLMContext("")
	convo.SetTools([]frames.Tool{{
		Name: "get_weather", Description: "Get the weather.", Parameters: json.RawMessage(`{"type":"object"}`),
		Handler: llm.FunctionCallHandler(func(ctx context.Context, p llm.FunctionCallParams) error {
			return p.Result(ctx, `{"temp":62,"conditions":"rain"}`, nil)
		}),
	}})
	backend, err := llmworker.NewBackend(llmworker.BackendConfig{
		Name: "backend", LLM: newScriptedLLM(runs...), Context: convo,
	})
	if err != nil {
		t.Fatal(err)
	}
	return backend
}

// runService runs the service in a pipeline worker under a runner, sends it a
// context frame, waits for the frames reaching the end of the pipeline to stop
// arriving, and returns them.
func runService(t *testing.T, service *LLMWithBackend, convo *frames.LLMContext) []frames.Frame {
	t.Helper()
	if convo == nil {
		return runSteps(t, service)
	}
	return runSteps(t, service, frames.NewLLMContextFrame(convo))
}

// runSteps runs the service as runService does, sending it each frame of steps
// in turn and sleeping for each duration, and returns the frames that reached
// the end of the pipeline.
func runSteps(t *testing.T, service *LLMWithBackend, steps ...any) []frames.Frame {
	t.Helper()
	ctx, cancel := context.WithTimeout(t.Context(), 15*time.Second)
	defer cancel()
	off := false
	worker := pipeline.NewWorker(pipeline.New(service), pipeline.WorkerConfig{
		Name:                    "frontend",
		ReachedDownstreamFilter: pipeline.AnyFrame,
		EnableRTVI:              &off,
		EnableTurnTracking:      &off,
		IdleTimeout:             -1,
	})
	var mu sync.Mutex
	var down []frames.Frame
	events.On(&worker.Registry, pipeline.EventFrameReachedDownstream, func(_ context.Context, f frames.Frame) {
		mu.Lock()
		down = append(down, f)
		mu.Unlock()
	})
	runner := workers.NewRunner(workers.RunnerConfig{HandleInterrupt: &off})
	runner.AddWorkers(ctx, worker)
	runDone := make(chan error, 1)
	go func() { runDone <- runner.Run(ctx, workers.RunOptions{}) }()

	for _, step := range steps {
		switch s := step.(type) {
		case time.Duration:
			time.Sleep(s)
		case frames.Frame:
			worker.QueueFrame(s)
		}
	}
	// Settle: stop once nothing new has arrived for a while.
	count := func() int {
		mu.Lock()
		defer mu.Unlock()
		return len(down)
	}
	for last, lastAt := count(), time.Now(); time.Since(lastAt) < 500*time.Millisecond; {
		if ctx.Err() != nil {
			t.Fatal("the pipeline was still busy when the test ran out of time")
		}
		time.Sleep(10 * time.Millisecond)
		if n := count(); n != last {
			last, lastAt = n, time.Now()
		}
	}
	runner.Cancel(context.Background(), "")
	<-runDone

	mu.Lock()
	defer mu.Unlock()
	return slices.Clone(down)
}

func TestClientTracingSendsEachExchangeAsAServerMessage(t *testing.T) {
	backend := weatherBackend(t,
		scriptedRun{calls: []frames.ToolCall{
			{ID: "c1", Name: "get_weather", Args: json.RawMessage(`{"location":"Seattle"}`)},
		}},
		scriptedRun{text: ">> It's 62 and raining."},
	)
	service, err := New(Config{
		Frontend: newDelegatingFrontend(""), Backend: backend,
		Connector: NewBackendConnector(ConnectorConfig{ClientTrace: true}),
	})
	if err != nil {
		t.Fatal(err)
	}

	down := runService(t, service, conversation(user("weather in seattle?")))

	var traced []map[string]any
	for _, f := range down {
		if sm, ok := f.(*rtvi.ServerMessageFrame); ok {
			if data, ok := sm.Data.(map[string]any); ok {
				traced = append(traced, data)
			}
		}
	}
	var kinds []string
	for _, d := range traced {
		if d["type"] != "llm-with-backend" {
			t.Errorf("traced %v", d)
		}
		kind, _ := d["event"].(string)
		kinds = append(kinds, kind)
	}
	want := []string{"request", "tool_call", "tool_call", "tool_call", "output", "idle"}
	if !slices.Equal(kinds, want) {
		t.Fatalf("traced %v, want %v", kinds, want)
	}
	request, _ := traced[0]["text"].(string)
	if traced[0]["backend"] != "idle" || !strings.Contains(request, "weather in seattle?") {
		t.Errorf("request traced as %v", traced[0])
	}
	if traced[4]["text"] != "It's 62 and raining." || traced[4]["spoken"] != true {
		t.Errorf("output traced as %v", traced[4])
	}
}

// TestALocalBackendIsHeardThroughTheFrontendsConversation runs the whole path,
// with the backend worker added and attached by the service itself.
func TestALocalBackendIsHeardThroughTheFrontendsConversation(t *testing.T) {
	backend := weatherBackend(t,
		scriptedRun{text: "Let me check.", calls: []frames.ToolCall{
			{ID: "c1", Name: "get_weather", Args: json.RawMessage(`{"location":"Seattle"}`)},
		}},
		scriptedRun{text: ">> It's 62 and raining."},
	)
	service, err := New(Config{Frontend: newDelegatingFrontend(""), Backend: backend})
	if err != nil {
		t.Fatal(err)
	}

	down := runService(t, service, conversation(user("weather in seattle?")))

	var results []*frames.FunctionCallResultFrame
	var appended []appendedMessage
	var started []*frames.ExternalFunctionCallsStartedFrame
	var reported []string
	for _, f := range down {
		switch fr := f.(type) {
		case *frames.FunctionCallResultFrame:
			results = append(results, fr)
		case *frames.LLMMessagesAppendFrame:
			appended = append(appended, appendedMessage{
				role: fr.Messages[0].Role, text: fr.Messages[0].Text, run: fr.RunLLM,
				interruptible: frames.Interruptible(fr),
			})
		case *frames.ExternalFunctionCallsStartedFrame:
			started = append(started, fr)
		case *frames.ExternalFunctionCallInProgressFrame:
			reported = append(reported, "in progress "+fr.ToolName)
		case *frames.ExternalFunctionCallResultFrame:
			reported = append(reported, "result "+fr.ToolName)
		}
	}

	if len(results) != 1 || results[0].Result != `{"backend":"idle","status":"delegated"}` ||
		results[0].Properties == nil || results[0].Properties.RunLLM == nil || !*results[0].Properties.RunLLM {
		t.Errorf("delegate settled with %+v", results)
	}
	// A message queued while the bot speaks waits behind the speech; an
	// interruption must not drop it on the way to the aggregator.
	//
	// The note and the record of the call are both silent, and come from two
	// places: the call is relayed as it passes the backend's aggregator, the note
	// when the turn's end is reported, which happens off that path. Either may
	// arrive first; the spoken result comes after both.
	silent := []appendedMessage{
		{frames.RoleDeveloper, "Backend (note): Let me check.", false, false},
		{frames.RoleDeveloper, "Backend (working): get_weather(location='Seattle')", false, false},
	}
	spoken := appendedMessage{
		frames.RoleDeveloper, "Backend: It's 62 and raining.\n\n" + BackendMessageRelayInstruction, true, false,
	}
	if len(appended) != 3 || !slices.Contains(appended[:2], silent[0]) || !slices.Contains(appended[:2], silent[1]) ||
		appended[2] != spoken {
		t.Errorf("appended %+v, want %+v in either order, then %+v", appended, silent, spoken)
	}
	// The backend's own call reached the frontend's pipeline as a report only.
	if len(started) != 1 || len(started[0].Calls) != 1 || started[0].Calls[0].ToolName != "get_weather" {
		t.Errorf("started %+v", started)
	}
	if !slices.Equal(reported, []string{"in progress get_weather", "result get_weather"}) {
		t.Errorf("reported %v", reported)
	}
}

// TestALocalBackendIsAddedBeforeTheFrontendStarts checks the backend worker is
// added before the frontend starts, since the frontend can delegate as soon as
// it gets the StartFrame.
func TestALocalBackendIsAddedBeforeTheFrontendStarts(t *testing.T) {
	backend := weatherBackend(t, scriptedRun{text: "Done."})
	frontend := newDelegatingFrontend("")
	var addedFirst bool
	frontend.onStart = func() { addedFirst = backend.Parent() != "" }
	service, err := New(Config{Frontend: frontend, Backend: backend})
	if err != nil {
		t.Fatal(err)
	}

	runService(t, service, nil)

	if !addedFirst {
		t.Error("the frontend started before the backend was added")
	}
}
