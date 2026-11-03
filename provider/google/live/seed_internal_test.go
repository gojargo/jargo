package live

import (
	"testing"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor"
)

// These tests are not ports of upstream tests: upstream has none for how a
// session is seeded (_create_initial_response and _handle_session_ready).

// clientContents are the clientContent messages the session received, in order.
func (r *recordingSession) clientContents() []map[string]any {
	return r.withKey("clientContent")
}

// textNudges counts the realtime text inputs that nudge a 3.x model to answer.
func (r *recordingSession) textNudges() int {
	n := 0
	for _, m := range r.withKey("realtimeInput") {
		if _, ok := m["text"]; ok {
			n++
		}
	}
	return n
}

// turnRoles are the roles of the turns a clientContent carries.
func turnRoles(content map[string]any) []string {
	turns, _ := content["turns"].([]any)
	roles := make([]string, 0, len(turns))
	for _, t := range turns {
		m, _ := t.(map[string]any)
		role, _ := m["role"].(string)
		roles = append(roles, role)
	}
	return roles
}

func greeting() *frames.LLMContext {
	convo := frames.NewLLMContext("")
	convo.AddUserMessage("Say hello.")
	return convo
}

// The first conversation seeds the session and, by default, has the model
// answer it; a 3.x model also needs a realtime input before it runs.
func TestTheFirstConversationSeedsTheSessionAndIsAnswered(t *testing.T) {
	srv := newRecordingSession(t)
	s := startedServiceBuilt(t, New(Config{APIKey: "k", BaseURL: wsURL(srv)}))

	s.handleContext(t.Context(), greeting(), processor.Downstream)
	settle()

	got := srv.clientContents()
	if len(got) != 1 || got[0][keyTurnComplete] != true {
		t.Fatalf("client content = %v, want one seed completing the turn", got)
	}
	if roles := turnRoles(got[0]); len(roles) != 1 || roles[0] != "user" {
		t.Errorf("seed roles = %v, want the user's turn", roles)
	}
	if n := srv.textNudges(); n != 1 {
		t.Errorf("sent %d realtime nudges, want one", n)
	}
	if !s.readyForInput.Load() {
		t.Error("the seeded session does not take input")
	}
}

// Told not to answer the first conversation, the model is seeded only.
func TestASeedNotToBeAnsweredDoesNotCompleteTheTurn(t *testing.T) {
	srv := newRecordingSession(t)
	no := false
	s := startedServiceBuilt(t, New(Config{
		APIKey: "k", BaseURL: wsURL(srv), InferenceOnContextInitialization: &no,
	}))

	s.handleContext(t.Context(), greeting(), processor.Downstream)
	settle()

	got := srv.clientContents()
	if len(got) != 1 || got[0][keyTurnComplete] != false {
		t.Fatalf("client content = %v, want one seed not completing the turn", got)
	}
	if n := srv.textNudges(); n != 0 {
		t.Errorf("sent %d realtime nudges, want none", n)
	}
}

// A 2.5 model needs the seed to end on a user turn, and, seeded without being
// asked to answer, has the user's first turn complete one.
func TestA25SeedEndsOnAUserTurnAndTheFirstTurnCompletesIt(t *testing.T) {
	srv := newRecordingSession(t)
	no := false
	s := startedServiceBuilt(t, New(Config{
		APIKey: "k", BaseURL: wsURL(srv), Model: "gemini-2.5-flash-native-audio-preview",
		InferenceOnContextInitialization: &no,
	}))
	convo := greeting()
	convo.AddAssistantMessage("Hello!")

	s.handleContext(t.Context(), convo, processor.Downstream)
	settle()

	got := srv.clientContents()
	if len(got) != 1 {
		t.Fatalf("client content = %v, want one seed", got)
	}
	if roles := turnRoles(got[0]); len(roles) != 3 || roles[1] != "model" || roles[2] != "user" {
		t.Errorf("seed roles = %v, want a blank user turn after the model's", roles)
	}

	s.handleUserStoppedSpeaking(t.Context())
	settle()
	got = srv.clientContents()
	if len(got) != 2 || got[1][keyTurnComplete] != true {
		t.Fatalf("client content = %v, want the first turn completed", got)
	}

	// Only the first turn.
	s.handleUserStoppedSpeaking(t.Context())
	settle()
	if n := len(srv.clientContents()); n != 2 {
		t.Errorf("sent %d client contents, want no second completion", n)
	}
}

// An empty first conversation with an instruction configured is not seeded,
// as upstream behaves: converting it against the instruction finds content
// (the instruction itself, put in as a user turn when there is no other text),
// so the instruction is not added for the model to answer, and converting it
// without the instruction to seed it finds nothing. The bot does not speak
// first. To have it greet, put a message in the conversation, as the upstream
// examples do.
func TestAnEmptyConversationWithAnInstructionIsNotSeeded(t *testing.T) {
	srv := newRecordingSession(t)
	s := startedServiceBuilt(t, New(Config{APIKey: "k", BaseURL: wsURL(srv), Instructions: "Greet the user."}))

	s.handleContext(t.Context(), frames.NewLLMContext(""), processor.Downstream)
	settle()

	if got := srv.clientContents(); len(got) != 0 {
		t.Fatalf("client content = %v, want none", got)
	}
	if !s.readyForInput.Load() {
		t.Error("the session does not take input")
	}
}

// An empty conversation with no instruction gives nothing to seed, so the
// session takes input right away.
func TestAnEmptyConversationWithoutAnInstructionTakesInputAtOnce(t *testing.T) {
	srv := newRecordingSession(t)
	s := startedServiceBuilt(t, New(Config{APIKey: "k", BaseURL: wsURL(srv)}))

	s.handleContext(t.Context(), frames.NewLLMContext(""), processor.Downstream)
	settle()

	if n := len(srv.clientContents()); n != 0 {
		t.Errorf("sent %d client contents, want none", n)
	}
	if !s.readyForInput.Load() {
		t.Error("the session does not take input")
	}
}

// A conversation arriving before the session is ready is seeded once it is.
func TestAConversationBeforeTheSessionIsReadyIsSeededWhenItIs(t *testing.T) {
	srv := newRecordingSession(t)
	s := startedServiceBuilt(t, New(Config{APIKey: "k", BaseURL: wsURL(srv)}))
	s.ready.Store(false)

	s.handleContext(t.Context(), greeting(), processor.Downstream)
	settle()
	if n := len(srv.clientContents()); n != 0 {
		t.Fatalf("sent %d client contents before the session was ready, want none", n)
	}

	s.handleSessionReady(t.Context())
	settle()
	if got := srv.clientContents(); len(got) != 1 || got[0][keyTurnComplete] != true {
		t.Fatalf("client content = %v, want the seed once ready", got)
	}
}

// A first conversation that reopens the session is seeded once, whichever of
// the reopened session becoming ready and the conversation being taken in
// comes first.
func TestAConversationThatReopensTheSessionIsSeededOnce(t *testing.T) {
	srv := newRecordingSession(t)
	s := startedServiceBuilt(t, New(Config{APIKey: "k", BaseURL: wsURL(srv)}))
	convo := greeting()
	convo.SetTools([]frames.Tool{sampleTool("from_context")})

	s.handleContext(t.Context(), convo, processor.Downstream)
	waitSetups(t, srv, 2)
	time.Sleep(200 * time.Millisecond)

	if got := srv.clientContents(); len(got) != 1 {
		t.Fatalf("client content = %v, want one seed", got)
	}
}

// A session opened again once the conversation is in is seeded again, without
// a 3.x model being asked to answer.
func TestAReopenedSessionIsSeededAgain(t *testing.T) {
	srv := newRecordingSession(t)
	s := startedServiceBuilt(t, New(Config{APIKey: "k", BaseURL: wsURL(srv)}))
	s.handleContext(t.Context(), greeting(), processor.Downstream)
	settle()

	s.handleSessionReady(t.Context())
	settle()

	got := srv.clientContents()
	if len(got) != 2 || got[1][keyTurnComplete] != false {
		t.Fatalf("client content = %v, want a second seed not completing the turn", got)
	}
}

// Audio is not sent until the session has been seeded.
func TestAudioWaitsForTheSessionToBeSeeded(t *testing.T) {
	srv := newRecordingSession(t)
	s := startedServiceBuilt(t, New(Config{APIKey: "k", BaseURL: wsURL(srv)}))

	s.sendAudio([]byte{1, 2, 3, 4}, 16000, 1)
	settle()
	if n := len(srv.withKey("realtimeInput")); n != 0 {
		t.Fatalf("sent %d realtime inputs before the session was seeded, want none", n)
	}

	s.handleContext(t.Context(), frames.NewLLMContext(""), processor.Downstream)
	s.sendAudio([]byte{1, 2, 3, 4}, 16000, 1)
	settle()
	if n := len(srv.withKey("realtimeInput")); n != 1 {
		t.Errorf("sent %d realtime inputs, want the audio", n)
	}
}

// Messages appended with no conversation are converted the way the session
// takes them, a leading system message being the prompt rather than a turn.
func TestAnAppendWithoutAConversationLiftsTheSystemMessage(t *testing.T) {
	srv := newRecordingSession(t)
	s := startedServiceBuilt(t, New(Config{APIKey: "k", BaseURL: wsURL(srv)}))

	f := frames.NewLLMMessagesAppendFrame([]frames.Message{
		{Role: frames.RoleSystem, Text: "Be brief."},
		{Role: frames.RoleUser, Text: "Hi."},
	})
	s.handleMessagesAppend(t.Context(), f)
	settle()

	got := srv.clientContents()
	if len(got) != 1 || got[0][keyTurnComplete] != true {
		t.Fatalf("client content = %v, want one completing the turn", got)
	}
	if roles := turnRoles(got[0]); len(roles) != 1 {
		t.Errorf("turns = %v, want the user's alone", roles)
	}
}

// The setup tells the API the history comes as client content, unless the
// connector's deployment does not take that.
func TestTheSetupCarriesTheHistoryConfig(t *testing.T) {
	setup := setupOf(t, New(Config{APIKey: "k"}))
	history, _ := setup["historyConfig"].(map[string]any)
	if history["initialHistoryInClientContent"] != true {
		t.Errorf("historyConfig = %v, want initial history in client content", history)
	}
	if _, ok := setupOf(t, NewWithConnector("Gated", noHistoryConnector{}, Config{}))["historyConfig"]; ok {
		t.Error("a connector without history configuration still got one")
	}
}

// noHistoryConnector is a deployment that takes no history configuration.
type noHistoryConnector struct{ gatedConnector }

func (noHistoryConnector) SupportsHistoryConfig() bool { return false }
