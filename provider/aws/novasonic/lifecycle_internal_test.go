package novasonic

import (
	"context"
	"encoding/json"
	"reflect"
	"strings"
	"testing"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/service/llm"
)

// (Not ports of upstream tests: upstream has none for the session's
// lifecycle, its output, or its resets.)

func TestDefaults(t *testing.T) {
	s := New(Config{})
	if got := s.Model(); got != "amazon.nova-2-5-sonic" {
		t.Errorf("model = %q, want amazon.nova-2-5-sonic", got)
	}
	if s.live.Voice.Or("") != "matthew" || s.live.Temperature.Or(0) != 0.7 ||
		s.live.MaxTokens.Or(0) != 1024 || s.live.TopP.Or(0) != 0.9 {
		t.Errorf("settings = %+v, want the defaults", s.live)
	}
	if s.audio != DefaultAudioConfig() {
		t.Errorf("audio = %+v, want the default", s.audio)
	}
}

// Setting up opens the session with the inference settings, and sets nothing
// more up until the first conversation arrives.
func TestSetupStartsTheSession(t *testing.T) {
	h := newHarness(t, Config{EndpointingSensitivity: EndpointingSensitivityHigh})
	st := h.client.stream(t, 0)
	if got := st.names(); !reflect.DeepEqual(got, []string{"sessionStart"}) {
		t.Fatalf("events = %v, want the session started only", got)
	}
	body := st.events()[0].body
	inference, _ := body["inferenceConfiguration"].(map[string]any)
	if inference["maxTokens"] != 1024.0 || inference["topP"] != 0.9 || inference["temperature"] != 0.7 {
		t.Errorf("inference = %v, want the defaults", inference)
	}
	turn, _ := body["turnDetectionConfiguration"].(map[string]any)
	if turn["endpointingSensitivity"] != "HIGH" {
		t.Errorf("turn detection = %v, want HIGH", turn)
	}
	if h.client.model != "amazon.nova-2-5-sonic" {
		t.Errorf("model = %q", h.client.model)
	}
}

// The first-generation model takes no endpointing sensitivity, so it is
// dropped.
func TestEndpointingSensitivityIsDroppedForTheFirstGenerationModel(t *testing.T) {
	s := New(Config{Model: firstGenerationModel, EndpointingSensitivity: EndpointingSensitivityLow})
	envelope, _ := s.sessionStartEvent()["event"].(map[string]any)
	body, _ := envelope["sessionStart"].(map[string]any)
	if _, ok := body["turnDetectionConfiguration"]; ok {
		t.Errorf("session start = %v, want no endpointing sensitivity", body)
	}
}

// The first conversation sets the session up: the tools, the system
// instruction, the history, the audio block, and the last user message as the
// one to answer.
func TestTheFirstConversationSetsTheSessionUp(t *testing.T) {
	h := newHarness(t, Config{Instructions: "Be brief.", Tools: sampleTools()})
	convo := frames.NewLLMContext("")
	convo.AddUserMessage("Hi")
	convo.AddAssistantMessage("Hello!")
	convo.AddUserMessage("What's up?")
	h.seed(t, convo)
	st := h.client.stream(t, 0)

	want := []string{
		"sessionStart", "promptStart",
		"contentStart", "textInput", "contentEnd", // the system instruction
		"contentStart", "textInput", "contentEnd", // Hi
		"contentStart", "textInput", "contentEnd", // Hello!
		"contentStart",                            // the audio block
		"contentStart", "textInput", "contentEnd", // What's up?
	}
	if got := st.names(); !reflect.DeepEqual(got, want) {
		t.Fatalf("events = %v, want %v", got, want)
	}
	texts := st.texts()
	wantTexts := []sentText{
		{role: "SYSTEM", text: "Be brief."},
		{role: "USER", text: "Hi"},
		{role: "ASSISTANT", text: "Hello!"},
		{role: "USER", text: "What's up?", interactive: true},
	}
	if !reflect.DeepEqual(texts, wantTexts) {
		t.Errorf("texts = %+v, want %+v", texts, wantTexts)
	}
	prompt := st.events()[1].body
	tools, _ := prompt["toolConfiguration"].(map[string]any)["tools"].([]any)
	if len(tools) != 1 {
		t.Errorf("prompt start = %v, want the configured tool declared", prompt)
	}
	audio := st.events()[11].body
	if audio["type"] != "AUDIO" || audio["role"] != "USER" || audio["interactive"] != true {
		t.Errorf("audio block = %v", audio)
	}
}

// A prompt start without tools declares none.
func TestAPromptStartWithoutToolsDeclaresNone(t *testing.T) {
	h, st := liveStream(t)
	_ = h
	prompt := st.events()[1].body
	if _, ok := prompt["toolConfiguration"]; ok {
		t.Errorf("prompt start = %v, want no tools", prompt)
	}
	out, _ := prompt["audioOutputConfiguration"].(map[string]any)
	if out["voiceId"] != "matthew" || out["sampleRateHertz"] != 24000.0 {
		t.Errorf("audio output = %v", out)
	}
}

// User audio reaches the session once its audio block is open, and travels on.
func TestUserAudioIsSentAndTravelsOn(t *testing.T) {
	h := newHarness(t, Config{})
	ctx := context.Background()
	audio := frames.NewInputAudioRawFrame([]byte{1, 2, 3, 4}, 16000, 1)
	if err := h.s.ProcessFrame(ctx, audio, processor.Downstream); err != nil {
		t.Fatal(err)
	}
	st := h.client.stream(t, 0)
	for _, n := range st.names() {
		if n == "audioInput" {
			t.Fatal("audio was sent before the audio block was open")
		}
	}
	h.seed(t, frames.NewLLMContext(""))
	if err := h.s.ProcessFrame(ctx, frames.NewInputAudioRawFrame([]byte{1, 2, 3, 4}, 16000, 1),
		processor.Downstream); err != nil {
		t.Fatal(err)
	}
	if got := st.names(); got[len(got)-1] != "audioInput" {
		t.Errorf("events = %v, want the audio sent", got)
	}
	n := 0
	for _, f := range h.down.frames() {
		if _, ok := f.(*frames.InputAudioRawFrame); ok {
			n++
		}
	}
	if n != 2 {
		t.Errorf("forwarded %d audio frames, want both", n)
	}
}

// A response opens on the assistant's first speculative text, closing the
// user's turn, and closes when its audio ends. The speculative text is shown as
// it comes and recorded once its audio has gone out.
func TestAResponseIsReportedLikeACascade(t *testing.T) {
	h, _ := liveStream(t)
	h.drive(t, "contentStart", contentStartBody("TEXT", "USER", "FINAL"))
	h.drive(t, "textOutput", textOutputBody("hello there"))
	h.drive(t, "contentEnd", contentEndBody("END_TURN"))

	h.drive(t, "contentStart", contentStartBody("TEXT", "ASSISTANT", "SPECULATIVE"))
	h.drive(t, "textOutput", textOutputBody("Hi!"))
	h.drive(t, "contentEnd", contentEndBody("PARTIAL_TURN"))
	h.drive(t, "contentStart", contentStartBody("AUDIO", "ASSISTANT", ""))
	h.drive(t, "audioOutput", map[string]any{"content": "AAAA"})
	h.drive(t, "contentEnd", contentEndBody("END_TURN"))

	want := []string{
		"LLMFullResponseStartFrame", "TTSStartedFrame",
		"LLMTextFrame", "AggregatedTextFrame",
		"TTSAudioRawFrame",
		"TTSTextFrame",
		"LLMFullResponseEndFrame", "TTSStoppedFrame",
	}
	if got := h.down.names(); !reflect.DeepEqual(got, want) {
		t.Errorf("downstream = %v, want %v", got, want)
	}
	var user *frames.TranscriptionFrame
	for _, f := range h.up.frames() {
		if tf, ok := f.(*frames.TranscriptionFrame); ok {
			user = tf
		}
	}
	if user == nil || user.Text != "hello there" {
		t.Errorf("user transcript = %+v, want it pushed upstream", user)
	}
	for _, f := range h.down.frames() {
		switch tf := f.(type) {
		case *frames.LLMTextFrame:
			if tf.AppendToContext {
				t.Error("the speculative text should be kept out of the conversation")
			}
		case *frames.TTSTextFrame:
			if tf.Text != "Hi!" || !tf.IncludesInterFrameSpaces {
				t.Errorf("tts text = %+v", tf)
			}
		}
	}
}

// The user's final text arrives in chunks, which are joined into one
// transcript when the turn closes.
func TestTheUsersTranscriptIsJoined(t *testing.T) {
	h, _ := liveStream(t)
	for _, chunk := range []string{"one", "two"} {
		h.drive(t, "contentStart", contentStartBody("TEXT", "USER", "FINAL"))
		h.drive(t, "textOutput", textOutputBody(chunk))
		h.drive(t, "contentEnd", contentEndBody("END_TURN"))
	}
	h.drive(t, "contentStart", contentStartBody("TEXT", "ASSISTANT", "SPECULATIVE"))
	var got []string
	for _, f := range h.up.frames() {
		if tf, ok := f.(*frames.TranscriptionFrame); ok {
			got = append(got, tf.Text)
		}
	}
	if !reflect.DeepEqual(got, []string{"one two"}) {
		t.Errorf("transcripts = %v, want one joined", got)
	}
}

// The audio cut off by the user is a barge-in: the interruption is broadcast
// ahead of the response's end.
func TestAudioInterruptedIsABargeIn(t *testing.T) {
	h, _ := liveStream(t)
	h.drive(t, "contentStart", contentStartBody("TEXT", "ASSISTANT", "SPECULATIVE"))
	h.drive(t, "contentStart", contentStartBody("AUDIO", "ASSISTANT", ""))
	h.drive(t, "contentEnd", contentEndBody("INTERRUPTED"))
	got := h.down.names()
	idx := func(name string) int {
		for i, n := range got {
			if n == name {
				return i
			}
		}
		return -1
	}
	if i, e := idx("InterruptionFrame"), idx("LLMFullResponseEndFrame"); i < 0 || e < 0 || i > e {
		t.Errorf("downstream = %v, want the interruption before the response's end", got)
	}
	up := false
	for _, f := range h.up.frames() {
		if _, ok := f.(*frames.InterruptionFrame); ok {
			up = true
		}
	}
	if !up {
		t.Error("the interruption should go upstream too")
	}
}

// The final text cut off is a barge-in too, which closes a response still open.
func TestTextInterruptedClosesTheResponse(t *testing.T) {
	h, _ := liveStream(t)
	h.drive(t, "contentStart", contentStartBody("TEXT", "ASSISTANT", "SPECULATIVE"))
	h.drive(t, "contentStart", contentStartBody("TEXT", "ASSISTANT", "FINAL"))
	h.drive(t, "contentEnd", contentEndBody("INTERRUPTED"))
	got := strings.Join(h.down.names(), ",")
	if !strings.Contains(got, "InterruptionFrame,LLMFullResponseEndFrame,TTSStoppedFrame") {
		t.Errorf("downstream = %v, want the interruption and the response closed", got)
	}
}

// A tool use runs the registered handler with the conversation.
func TestToolUseRunsTheHandler(t *testing.T) {
	h, _ := liveStream(t)
	called := make(chan json.RawMessage, 1)
	h.s.RegisterFunction("get_weather", llm.FunctionCallHandler(
		func(ctx context.Context, p llm.FunctionCallParams) error {
			called <- p.Arguments
			return p.Result(ctx, weather, nil)
		}))
	h.drive(t, "contentStart", contentStartBody("TOOL", "TOOL", ""))
	h.drive(t, "toolUse", map[string]any{
		"toolName": "get_weather", "toolUseId": "call_1", "content": `{"city":"Paris"}`,
	})
	select {
	case args := <-called:
		if string(args) != `{"city":"Paris"}` {
			t.Errorf("arguments = %s", args)
		}
	case <-waitTimeout():
		t.Fatal("the handler did not run")
	}
}

// A tool use naming a function no handler claims is an error.
func TestAnUnhandledToolUseIsAnError(t *testing.T) {
	h, _ := liveStream(t)
	h.drive(t, "contentStart", contentStartBody("TOOL", "TOOL", ""))
	raw, _ := json.Marshal(map[string]any{"toolName": "nope", "toolUseId": "c", "content": "{}"})
	err := h.s.handleEvent(context.Background(), map[string]json.RawMessage{"toolUse": raw})
	var unhandled *unhandledFunctionError
	if err == nil || !errorsAs(err, &unhandled) {
		t.Errorf("err = %v, want the function reported unhandled", err)
	}
}

// A tool result is sent as a block of its own naming the call it answers, once;
// a placeholder for a call still running or canceled is not sent.
func TestToolResultsFromTheConversation(t *testing.T) {
	h, st := liveStream(t)
	convo := frames.NewLLMContext("")
	convo.AddMessage(frames.Message{Role: frames.RoleUser, ToolResults: []frames.ToolResult{
		{ID: "running", Content: frames.ToolResultInProgress},
		{ID: "canceled", Content: frames.ToolResultCanceled},
		{ID: "done", Content: weather},
	}})
	// The service keeps the first conversation; a later report carries the
	// results.
	h.s.mu.Lock()
	h.s.convo = convo
	h.s.mu.Unlock()
	h.seed(t, convo)
	h.seed(t, convo)
	if n := st.toolResults(); n != 1 {
		t.Fatalf("tool results sent = %d, want the finished one once", n)
	}
	var start, result map[string]any
	for _, e := range st.events() {
		if e.name == "contentStart" && e.body["type"] == "TOOL" {
			start = e.body
		}
		if e.name == "toolResult" {
			result = e.body
		}
	}
	cfg, _ := start["toolResultInputConfiguration"].(map[string]any)
	if cfg["toolUseId"] != "done" || start["role"] != "TOOL" || result[keyContent] != weather {
		t.Errorf("block = %v / %v", start, result)
	}
}

// A session that fails while it is still wanted is reported, and the
// conversation is reset onto a new session seeded with it.
func TestAFailedSessionIsReset(t *testing.T) {
	h := newHarness(t, Config{})
	convo := frames.NewLLMContext("")
	convo.AddUserMessage("Hello")
	h.seed(t, convo)
	h.client.stream(t, 0).end()
	second := h.client.stream(t, 1)
	waitFor(t, func() bool {
		for _, tx := range second.texts() {
			if tx.text == "Hello" && tx.interactive {
				return true
			}
		}
		return false
	})
	var errs []string
	for _, f := range h.up.frames() {
		if e, ok := f.(*frames.ErrorFrame); ok {
			errs = append(errs, e.Error)
		}
	}
	if len(errs) != 1 || !strings.Contains(errs[0], "ended the response stream") {
		t.Errorf("errors = %v, want the stream ending reported", errs)
	}
}

// Ending the pipeline closes the prompt and the session.
func TestEndingClosesTheSession(t *testing.T) {
	h, st := liveStream(t)
	if err := h.s.ProcessFrame(context.Background(), frames.NewEndFrame(), processor.Downstream); err != nil {
		t.Fatal(err)
	}
	got := st.names()
	if n := len(got); n < 2 || got[n-2] != "promptEnd" || got[n-1] != "sessionEnd" {
		t.Errorf("events = %v, want the prompt and the session closed", got)
	}
}

// Settings changed at runtime reach the next session, which is said.
func TestSettingsUpdatesAreNotApplied(t *testing.T) {
	s := New(Config{})
	if s.Settings() != any(s.live) {
		t.Error("the settings store is the service's")
	}
}
