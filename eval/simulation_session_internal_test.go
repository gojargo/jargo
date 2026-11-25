package eval

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"reflect"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/gojargo/jargo/audio/vad"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor/rtvi"
	"github.com/gojargo/jargo/processor/turns"
	"github.com/gojargo/jargo/service/llm"
	"github.com/gojargo/jargo/service/stt"
)

// A simulation end to end: the persona LLM in the pipeline talks to the fake
// bot.

// endCallReply is a scripted persona reply that calls end_call with its
// arguments, the way a real model would.
type endCallReply map[string]any

// scriptedPersonaGen answers the bot's last message from a script: a reply's
// text, or an endCallReply.
type scriptedPersonaGen struct {
	script map[string]any

	mu   sync.Mutex
	seen []string
}

func (*scriptedPersonaGen) Generate(context.Context, *frames.LLMContext, llm.Emit) error { return nil }

func (g *scriptedPersonaGen) GenerateWithTools(ctx context.Context, convo *frames.LLMContext, sink llm.Sink) error {
	messages := convo.Messages()
	last := ""
	for _, m := range slices.Backward(messages) {
		if m.Role == frames.RoleUser {
			last = m.Text
			break
		}
	}
	g.mu.Lock()
	g.seen = append(g.seen, last)
	g.mu.Unlock()
	slog.InfoContext(ctx, "scripted persona: answering "+last)
	switch reply := g.script[last].(type) {
	case string:
		return sink.Text(reply)
	case endCallReply:
		args, _ := json.Marshal(map[string]any(reply))
		return sink.Tool(frames.ToolCall{ID: "c1", Name: EndCallFunction, Args: args})
	}
	return nil
}

func (g *scriptedPersonaGen) heard() []string {
	g.mu.Lock()
	defer g.mu.Unlock()
	return slices.Clone(g.seen)
}

// scriptedPersona is a persona LLM service answering from script, and its
// generator.
func scriptedPersona(script map[string]any) (*llm.Base, *scriptedPersonaGen) {
	gen := &scriptedPersonaGen{script: script}
	return llm.New("ScriptedPersonaLLM", gen), gen
}

// yesJudge says yes to the goal and to every turn on every criterion, and
// records what it saw.
func yesJudge() *fakeConversationJudge { return newFakeConversationJudge([]string{VerdictYes}) }

func runSimulationSession(t *testing.T, s *SimulationSession) *SimulationResult {
	t.Helper()
	res, err := s.Run(t.Context())
	if err != nil {
		t.Fatal(err)
	}
	return res
}

func TestSimulationIntegrationTextSimulationRunsToEndCall(t *testing.T) {
	bot := newFakeRTVIServer(t)
	bot.greeting = reply("Hello! How can I help?")
	bot.onText("What is the capital of Germany?", reply("The capital of Germany is Berlin.")...)
	persona, _ := scriptedPersona(map[string]any{
		"Hello! How can I help?":            "What is the capital of Germany?",
		"The capital of Germany is Berlin.": endCallReply{"success": true, "reason": "I learned it"},
	})
	simulation := testSimulation(func(s *SimulationScenario) {
		s.Persona, s.Success, s.MaxTurns, s.MaxDurationS = "A curious traveler.", "the bot named Berlin", 5, 10
	}, withMetrics(SimulationMetric{Name: "politeness", Criterion: "stayed polite"}))
	judge := yesJudge()
	session, err := NewSimulationSession(simulation, bot.url(), SessionParams{}, persona, judge, nil, nil)
	if err != nil {
		t.Fatal(err)
	}

	res := runSimulationSession(t, session)

	if res.Error != "" || res.FailureKind != "" {
		t.Fatalf("result = %+v\n%s", res, strings.Join(res.DebugLog, "\n"))
	}
	if !res.Succeeded || res.EndedBy != SimulationEndedByEndCall {
		t.Fatalf("result = %+v", res)
	}
	if want := map[string]any{"success": true, "reason": "I learned it"}; !reflect.DeepEqual(res.EndCall, want) {
		t.Fatalf("end call = %v", res.EndCall)
	}
	// The question; end_call is not a turn.
	if res.Turns != 1 {
		t.Fatalf("turns = %d", res.Turns)
	}
	// The persona's question went to the bot as one text turn, not spoken.
	sent := bot.messages(rtvi.TypeSendText)
	if got := bot.sentTexts(); !slices.Equal(got, []string{"What is the capital of Germany?"}) {
		t.Fatalf("sent %v", got)
	}
	var data rtvi.SendTextData
	_ = json.Unmarshal(sent[0].Data, &data)
	if data.Options == nil || data.Options.AudioResponse == nil || *data.Options.AudioResponse {
		t.Fatalf("send-text options = %+v, want no audio response", data.Options)
	}
	// The judge saw the whole conversation, persona as the user.
	want := []map[string]any{
		{"role": "assistant", "content": "Hello! How can I help?"},
		{"role": "user", "content": "What is the capital of Germany?"},
		{"role": "assistant", "content": "The capital of Germany is Berlin."},
	}
	if !reflect.DeepEqual(res.Messages, want) {
		t.Fatalf("messages = %v", res.Messages)
	}
	if !slices.Equal(judge.criteria, []string{"the bot named Berlin"}) ||
		!reflect.DeepEqual(judge.runCriteria, []Criterion{{"politeness", "stayed polite"}}) {
		t.Fatalf("judged %v on %v", judge.criteria, judge.runCriteria)
	}
	// Two bot turns, the greeting and the answer, reached the judge.
	turns := 0
	for _, e := range judge.seen() {
		if e["role"] == "assistant" {
			turns++
		}
	}
	if turns != 2 {
		t.Fatalf("judge transcript = %v", judge.seen())
	}
	// The bot was asked for its calls with their arguments, the judge's
	// evidence.
	configured := bot.evalMessages("eval-configure")
	full := map[string]any{"*": "full"}
	if len(configured) != 1 || !reflect.DeepEqual(configured[0]["function_call_report_level"], full) {
		t.Fatalf("eval-configure = %v", configured)
	}
}

func TestSimulationIntegrationASimulationKeepsTheKindOfTheFailureThatEndedIt(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		http.Error(w, "forbidden", http.StatusForbidden)
	}))
	defer srv.Close()
	persona, _ := scriptedPersona(nil)
	simulation := testSimulation(func(s *SimulationScenario) { s.MaxTurns = 1 })
	session, err := NewSimulationSession(simulation, wsURL(srv.URL), SessionParams{}, persona, yesJudge(), nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	res := runSimulationSession(t, session)
	if !strings.Contains(res.Error, "403") || res.FailureKind != FailureConnectFailed {
		t.Fatalf("result = %+v", res)
	}
}

func TestSimulationIntegrationAHarnessErrorWhileJudgingFailsTheRun(t *testing.T) {
	// The conversation is over; the caller's voice fails as the judge decides.
	bot := newFakeRTVIServer(t)
	bot.greeting = reply("Hello!")
	persona, _ := scriptedPersona(map[string]any{"Hello!": endCallReply{"success": true, "reason": "ok"}})
	judge := yesJudge()
	simulation := testSimulation(func(s *SimulationScenario) { s.MaxTurns, s.MaxDurationS = 3, 10 })
	session, err := NewSimulationSession(simulation, bot.url(), SessionParams{}, persona, judge, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	judge.evaluated = func() { session.stream.HarnessFailed("the caller voice failed") }

	res := runSimulationSession(t, session)

	if res.Succeeded || res.Error != "the caller voice failed" || res.FailureKind != FailureError {
		t.Fatalf("result = %+v", res)
	}
}

func TestSimulationIntegrationTheBotHangingUpEndsTheRun(t *testing.T) {
	bot := newFakeRTVIServer(t)
	bot.greeting = reply("How many in your party?")
	bot.onText("Two, please.", reply("Booked. Goodbye!")...)
	bot.closeAfter = "Two, please."
	persona, _ := scriptedPersona(map[string]any{
		"How many in your party?": "Two, please.", "Booked. Goodbye!": "Thanks!",
	})
	simulation := testSimulation(func(s *SimulationScenario) { s.MaxTurns, s.MaxDurationS = 5, 10 })
	session, err := NewSimulationSession(simulation, bot.url(), SessionParams{}, persona, yesJudge(), nil, nil)
	if err != nil {
		t.Fatal(err)
	}

	res := runSimulationSession(t, session)

	if res.Error != "" || res.EndedBy != SimulationEndedByBot || !res.Succeeded {
		t.Fatalf("result = %+v\n%s", res, strings.Join(res.DebugLog, "\n"))
	}
}

// A refused simulation is an error the persona never answers.
func TestAudioSupportCheckARefusedSimulationIsAnErrorThePersonaNeverAnswers(t *testing.T) {
	// The bot greets right after bot-ready, while the refused run is torn down.
	bot := newFakeRTVIServer(t)
	bot.capabilities = map[string]any{"audio_in": false, "audio_out": true}
	bot.greeting = reply("Welcome!")
	persona, gen := scriptedPersona(map[string]any{"Welcome!": "A table for two, please."})
	simulation := testSimulation(func(s *SimulationScenario) { s.MaxTurns, s.MaxDurationS = 5, 10 })
	session, err := NewSimulationSession(simulation, bot.url(),
		SessionParams{AudioSupportCheck: AudioSupportCheckWhenAdvertised}, persona, yesJudge(), failingVoice(t), nil)
	if err != nil {
		t.Fatal(err)
	}

	res := runSimulationSession(t, session)

	if res.Succeeded || !strings.Contains(res.Error, "audio_in_enabled") || res.FailureKind != FailureAudioUnsupported {
		t.Fatalf("result = %+v", res)
	}
	if got := gen.heard(); len(got) != 0 {
		t.Fatalf("the persona answered %v", got)
	}
}

// SessionFromScenario builds the session of the scenario's kind.
func TestSessionFromScenario(t *testing.T) {
	script := func() *ScriptScenario {
		return &ScriptScenario{Name: "capital", Turns: []ScriptTurn{
			{User: "hi", Expect: []Expectation{{Event: EventLLMStarted}}},
		}}
	}
	simulation := func() *SimulationScenario {
		return testSimulation(func(s *SimulationScenario) { s.Success = "the bot named Berlin" })
	}
	persona := func() PersonaLLM { p, _ := scriptedPersona(nil); return p }

	t.Run("a script scenario gets a script session", func(t *testing.T) {
		session, err := SessionFromScenario(script(), "ws://localhost:0", SessionParams{}, SessionServices{})
		if err != nil {
			t.Fatal(err)
		}
		if _, ok := session.(*ScriptSession); !ok {
			t.Fatalf("session = %T", session)
		}
	})
	t.Run("a simulation gets a simulation session", func(t *testing.T) {
		session, err := SessionFromScenario(simulation(), "ws://localhost:0", SessionParams{},
			SessionServices{PersonaLLM: persona(), Judge: yesJudge()})
		if err != nil {
			t.Fatal(err)
		}
		if _, ok := session.(*SimulationSession); !ok {
			t.Fatalf("session = %T", session)
		}
	})
	t.Run("params reach the driver", func(t *testing.T) {
		params := SessionParams{DefaultTimeoutMS: 1234}
		session, err := SessionFromScenario(script(), "ws://localhost:0", params, SessionServices{})
		if err != nil {
			t.Fatal(err)
		}
		s, ok := session.(*ScriptSession)
		if !ok {
			t.Fatalf("session = %T", session)
		}
		if s.Driver().defaultTimeoutMS != 1234 {
			t.Fatalf("default timeout = %d", s.Driver().defaultTimeoutMS)
		}
	})
	t.Run("a persona LLM is rejected for a script", func(t *testing.T) {
		_, err := SessionFromScenario(script(), "ws://localhost:0", SessionParams{},
			SessionServices{PersonaLLM: persona()})
		if !errors.Is(err, errPersonaForScript) {
			t.Fatalf("err = %v", err)
		}
	})
	t.Run("a judge that cannot judge a run is rejected for a simulation", func(t *testing.T) {
		_, err := SessionFromScenario(simulation(), "ws://localhost:0", SessionParams{},
			SessionServices{PersonaLLM: persona(), Judge: newFakeJudge()})
		if !errors.Is(err, errNotARunJudge) {
			t.Fatalf("err = %v", err)
		}
	})
	t.Run("other scenarios are rejected", func(t *testing.T) {
		rubric, _ := NewRubric("booking", "the bot booked a table", nil)
		_, err := SessionFromScenario(rubric, "ws://localhost:0", SessionParams{}, SessionServices{})
		if !errors.Is(err, errNotRunnable) {
			t.Fatalf("err = %v", err)
		}
	})
}

// A simulation built from its scenario builds the persona LLM its simulator:
// block names.
func TestSimulationSessionFromScenarioBuildsThePersonaLLM(t *testing.T) {
	session, err := SimulationSessionFromScenario(testSimulation(), "ws://localhost:0", SessionParams{},
		nil, yesJudge(), nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	if session.client.persona == nil || session.client.persona.LLM() == nil {
		t.Fatal("no persona LLM built")
	}
	_, err = NewSimulationSession(testSimulation(), "ws://localhost:0", SessionParams{}, nil, nil, nil, nil)
	if !errors.Is(err, errNoPersonaLLM) {
		t.Fatalf("err = %v, want a persona LLM required", err)
	}
}

// sequenceTranscriber hears each of its texts in turn, one per segment.
type sequenceTranscriber struct {
	mu    sync.Mutex
	texts []string
}

func (s *sequenceTranscriber) Transcribe(context.Context, []byte, int) (string, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if len(s.texts) == 0 {
		return "", nil
	}
	text := s.texts[0]
	s.texts = s.texts[1:]
	return text, nil
}

// spoken is an LLM response's messages followed by half a second of the bot's
// speech.
func spoken(text string) []rtvi.Message {
	replies := reply(text)
	speech := bytes.Repeat([]byte{0x20, 0x10}, 16000/2)
	for off := 0; off < len(speech); off += 1280 {
		replies = append(replies, botAudioMessage(speech[off:min(off+1280, len(speech))], 16000))
	}
	return replies
}

// spokenAfter is spoken, opened by a pause of silence before the bot speaks, as
// a bot takes a moment to answer: it lets the turn before it end first.
func spokenAfter(pause time.Duration, text string) []rtvi.Message {
	head := reply(text)
	silence := make([]byte, int(pause.Seconds()*16000)*2)
	for off := 0; off < len(silence); off += 1280 {
		head = append(head, botAudioMessage(silence[off:min(off+1280, len(silence))], 16000))
	}
	return append(head, spoken(text)[len(reply(text)):]...)
}

// In audio mode the persona hears the bot's transcribed speech, handed over by
// the turn aggregator, and answers it.
func TestSimulationIntegrationThePersonaAnswersTheBotsSpeech(t *testing.T) {
	bot := newFakeRTVIServer(t)
	bot.greeting = spoken("How can I help?")
	bot.onText("A table for two.", spokenAfter(time.Second, "Booked.")...)
	persona, gen := scriptedPersona(map[string]any{
		"How can I help?": "A table for two.",
		"Booked.":         endCallReply{"success": true, "reason": "booked"},
	})
	simulation := testSimulation(func(s *SimulationScenario) { s.BotAudio, s.MaxTurns, s.MaxDurationS = true, 5, 20 })
	transcriber := &sequenceTranscriber{texts: []string{"How can I help?", "Booked."}}
	session, err := NewSimulationSession(simulation, bot.url(), SessionParams{}, persona, yesJudge(), nil,
		stt.NewSegment("BotSTT", transcriber, 0))
	if err != nil {
		t.Fatal(err)
	}
	session.client.newBotVAD = func() (vad.Analyzer, error) {
		params := vad.DefaultParams()
		params.StopSecs = botVADStopSecs
		return &energyVAD{params: params}, nil
	}
	// A speech timeout hears the bot stop rather than the default Smart Turn
	// model, which needs the ONNX runtime. The turn then ends once the
	// persona's LLM marks it complete.
	session.client.botTurnStop = []turns.StopStrategy{
		turns.NewSpeechTimeoutStop(turns.SpeechTimeoutConfig{UserSpeechTimeout: 200 * time.Millisecond}),
	}

	res := runSimulationSession(t, session)

	if res.Error != "" || res.EndedBy != SimulationEndedByEndCall || res.Turns != 1 {
		t.Fatalf("result = %+v\n%s", res, strings.Join(res.DebugLog, "\n"))
	}
	want := []map[string]any{
		{"role": "assistant", "content": "How can I help?"},
		{"role": "user", "content": "A table for two."},
		{"role": "assistant", "content": "Booked."},
	}
	if !reflect.DeepEqual(res.Messages, want) {
		t.Fatalf("messages = %v\n%s", res.Messages, strings.Join(res.DebugLog, "\n"))
	}
	if got := gen.heard(); !slices.Equal(got, []string{"How can I help?", "Booked."}) {
		t.Fatalf("the persona heard %v", got)
	}
	if got := bot.sentTexts(); !slices.Equal(got, []string{"A table for two."}) {
		t.Fatalf("sent %v", got)
	}
}
