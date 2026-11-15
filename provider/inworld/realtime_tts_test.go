package inworld

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"reflect"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/coder/websocket"
	ttsadapter "github.com/gojargo/jargo/adapter/tts"
	ttsinworld "github.com/gojargo/jargo/adapter/tts/inworld"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/language"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/service/tts"
	"github.com/gojargo/jargo/service/wsservice"
	"github.com/gojargo/jargo/service/wsutil"
	"github.com/gojargo/jargo/utils/events"
)

// turns are the utterances the connection tests speak, one TTSSpeakFrame each.
//
//nolint:gochecknoglobals // test fixture
var turns = []string{"Hello there.", "How are you?", "Goodbye."}

// fakeInworld is a local Inworld WebSocket server. It records every message
// each connection receives and, unless silent, answers the way Inworld does:
// a create with contextCreated, text with audio, a flush with flushCompleted
// and a close with contextClosed. It can drop its first connection after a
// number of contexts have closed, and delay every handshake after the first.
type fakeInworld struct {
	t *testing.T
	// silent records what is sent and answers nothing.
	silent bool
	// dropAfterTurns aborts the first connection once this many of its
	// contexts have closed.
	dropAfterTurns int
	// handshakeDelay delays every handshake after the first.
	handshakeDelay time.Duration

	url string

	mu         sync.Mutex
	handshakes int
	messages   [][]map[string]any
}

func newFakeInworld(t *testing.T, f *fakeInworld) *fakeInworld {
	t.Helper()
	f.t = t
	srv := httptest.NewServer(http.HandlerFunc(f.serve))
	t.Cleanup(srv.Close)
	f.url = "ws" + strings.TrimPrefix(srv.URL, "http")
	return f
}

func (f *fakeInworld) serve(w http.ResponseWriter, r *http.Request) {
	f.mu.Lock()
	f.handshakes++
	delay := time.Duration(0)
	if f.handshakes > 1 {
		delay = f.handshakeDelay
	}
	f.mu.Unlock()
	time.Sleep(delay)

	c, err := websocket.Accept(w, r, nil)
	if err != nil {
		return
	}
	defer func() { _ = c.CloseNow() }()

	f.mu.Lock()
	index := len(f.messages)
	f.messages = append(f.messages, nil)
	f.mu.Unlock()

	ctx := r.Context()
	send := func(contextID any, result map[string]any) {
		result["contextId"] = contextID
		payload, _ := json.Marshal(map[string]any{"result": result})
		_ = c.Write(ctx, websocket.MessageText, payload)
	}
	status := func(contextID any, code int, message string) {
		send(contextID, map[string]any{"status": map[string]any{"code": code, "message": message}})
	}
	contexts := map[string]bool{}
	closed := 0
	for {
		_, data, err := c.Read(ctx)
		if err != nil {
			return
		}
		var msg map[string]any
		if err := json.Unmarshal(data, &msg); err != nil {
			f.t.Errorf("decoding a client message: %v", err)
			return
		}
		f.mu.Lock()
		f.messages[index] = append(f.messages[index], msg)
		f.mu.Unlock()
		if f.silent {
			continue
		}
		contextID, _ := msg["contextId"].(string)
		switch {
		case msg["create"] != nil:
			contexts[contextID] = true
			send(contextID, map[string]any{"contextCreated": map[string]any{}})
		case msg["send_text"] != nil:
			text, _ := msg["send_text"].(map[string]any)["text"].(string)
			switch {
			case msg["contextId"] == nil:
				status(nil, 3, "context_id is required")
			case !contexts[contextID]:
				status(contextID, 5, "Context not found")
			case text != "":
				audio := base64.StdEncoding.EncodeToString(make([]byte, 960))
				send(contextID, map[string]any{"audioChunk": map[string]any{"audioContent": audio}})
			}
		case msg["flush_context"] != nil:
			send(contextID, map[string]any{"flushCompleted": map[string]any{}})
		case msg["close_context"] != nil:
			delete(contexts, contextID)
			send(contextID, map[string]any{"contextClosed": map[string]any{}})
			closed++
			if index == 0 && closed == f.dropAfterTurns {
				// Gone without a closing handshake, as a dropped network is.
				_ = c.CloseNow()
				return
			}
		}
	}
}

// handshakeCount is how many connections were attempted.
func (f *fakeInworld) handshakeCount() int {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.handshakes
}

// sent is every message received, across connections.
func (f *fakeInworld) sent() []map[string]any {
	f.mu.Lock()
	defer f.mu.Unlock()
	var out []map[string]any
	for _, conn := range f.messages {
		out = append(out, conn...)
	}
	return out
}

// textsSent is the non-empty text sent on one connection, trimmed.
func (f *fakeInworld) textsSent(index int) []string {
	f.mu.Lock()
	defer f.mu.Unlock()
	var out []string
	if index >= len(f.messages) {
		return out
	}
	for _, m := range f.messages[index] {
		st, ok := m["send_text"].(map[string]any)
		if !ok {
			continue
		}
		if text, _ := st["text"].(string); text != "" {
			out = append(out, strings.TrimSpace(text))
		}
	}
	return out
}

// waitForMessages waits until the server has received n messages.
func (f *fakeInworld) waitForMessages(t *testing.T, n int) []map[string]any {
	t.Helper()
	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		if got := f.sent(); len(got) >= n {
			return got
		}
		time.Sleep(5 * time.Millisecond)
	}
	got := f.sent()
	t.Fatalf("the server received %d messages, want %d: %v", len(got), n, got)
	return nil
}

// kinds names each message by its key other than the context id.
func kinds(sent []map[string]any) []string {
	out := make([]string, 0, len(sent))
	for _, m := range sent {
		for k := range m {
			if k != "contextId" {
				out = append(out, k)
				break
			}
		}
	}
	return out
}

// contextIDs is the set of context ids the messages name.
func contextIDs(sent []map[string]any) map[string]bool {
	out := map[string]bool{}
	for _, m := range sent {
		if id, ok := m["contextId"].(string); ok {
			out[id] = true
		}
	}
	return out
}

// testRealtimeTTS builds the service against the fake server.
func testRealtimeTTS(f *fakeInworld, cfg RealtimeTTSConfig) (*tts.Base, *realtimeSynthesizer) {
	cfg.APIKey = "test-key"
	cfg.URL = f.url
	return newRealtimeTTS(cfg)
}

// createConfig builds the service, has it create a context, and returns the
// create configuration it sent.
func createConfig(t *testing.T, cfg RealtimeTTSConfig) map[string]any {
	t.Helper()
	f := newFakeInworld(t, &fakeInworld{silent: true})
	_, s := testRealtimeTTS(f, cfg)
	ctx := context.Background()
	s.Connect(ctx)
	t.Cleanup(func() { s.Disconnect(ctx) })
	if err := s.sendContext(ctx, "ctx-1"); err != nil {
		t.Fatalf("send context: %v", err)
	}
	msg := f.waitForMessages(t, 1)[0]
	create, ok := msg["create"].(map[string]any)
	if !ok {
		t.Fatalf("the message sent is not a create: %v", msg)
	}
	return create
}

func TestRealtimeTTSTokenModeUsesSentenceBoundary(t *testing.T) {
	config := createConfig(t, RealtimeTTSConfig{TextAggregation: frames.TextTypeToken})
	if config["autoMode"] != true {
		t.Errorf("autoMode = %v, want true", config["autoMode"])
	}
	if config["autoModeStrategy"] != "SENTENCE_BOUNDARY" {
		t.Errorf("autoModeStrategy = %v, want SENTENCE_BOUNDARY", config["autoModeStrategy"])
	}
	if _, ok := config["bufferCharThreshold"]; ok {
		t.Errorf("bufferCharThreshold = %v, want it left to the server", config["bufferCharThreshold"])
	}
}

func TestRealtimeTTSSentenceBoundarySendsASetBufferCharThreshold(t *testing.T) {
	config := createConfig(t, RealtimeTTSConfig{TextAggregation: frames.TextTypeToken, BufferCharThreshold: 400})
	if config["autoModeStrategy"] != "SENTENCE_BOUNDARY" {
		t.Errorf("autoModeStrategy = %v, want SENTENCE_BOUNDARY", config["autoModeStrategy"])
	}
	if config["bufferCharThreshold"] != float64(400) {
		t.Errorf("bufferCharThreshold = %v, want 400", config["bufferCharThreshold"])
	}
}

// The buffer settings configured are the ones sent.
func TestRealtimeTTSBufferSettingsAreSent(t *testing.T) {
	config := createConfig(t, RealtimeTTSConfig{MaxBufferDelayMs: 500, BufferCharThreshold: 100})
	if config["maxBufferDelayMs"] != float64(500) {
		t.Errorf("maxBufferDelayMs = %v, want 500", config["maxBufferDelayMs"])
	}
	if config["bufferCharThreshold"] != float64(100) {
		t.Errorf("bufferCharThreshold = %v, want 100", config["bufferCharThreshold"])
	}
}

func TestRealtimeTTSSentenceBoundaryNotRequested(t *testing.T) {
	for name, cfg := range map[string]RealtimeTTSConfig{
		"sentence aggregation": {TextAggregation: frames.TextTypeSentence},
		"an older model":       {TextAggregation: frames.TextTypeToken, Model: "inworld-tts-1.5-max"},
		"auto mode off":        {TextAggregation: frames.TextTypeToken, AutoMode: new(false)},
	} {
		t.Run(name, func(t *testing.T) {
			config := createConfig(t, cfg)
			if _, ok := config["autoModeStrategy"]; ok {
				t.Errorf("autoModeStrategy = %v, want none", config["autoModeStrategy"])
			}
			if config["bufferCharThreshold"] != float64(250) {
				t.Errorf("bufferCharThreshold = %v, want 250", config["bufferCharThreshold"])
			}
		})
	}
}

// The context is created with the first text it is given, and every text after
// goes on that context.
func TestRealtimeTTSContextIsCreatedWithFirstText(t *testing.T) {
	f := newFakeInworld(t, &fakeInworld{silent: true})
	b, s := testRealtimeTTS(f, RealtimeTTSConfig{})
	ctx := context.Background()
	t.Cleanup(func() { s.Disconnect(ctx) })

	b.CreateAudioContext("ctx-1")
	for _, text := range []string{"Hello there. ", "How are you? "} {
		if err := s.RunTTS(ctx, text, "ctx-1", nil); err != nil {
			t.Fatalf("run tts: %v", err)
		}
	}

	sent := f.waitForMessages(t, 3)
	if got, want := kinds(sent), []string{"create", "send_text", "send_text"}; !slices.Equal(got, want) {
		t.Errorf("messages = %v, want %v", got, want)
	}
	if ids := contextIDs(sent); len(ids) != 1 || !ids["ctx-1"] {
		t.Errorf("contexts = %v, want only ctx-1", ids)
	}
}

// A turn whose text never reached the service created no context, so
// completing it sends nothing.
func TestRealtimeTTSTurnWithoutTextSendsNothing(t *testing.T) {
	f := newFakeInworld(t, &fakeInworld{silent: true})
	_, s := testRealtimeTTS(f, RealtimeTTSConfig{})
	ctx := context.Background()
	s.Connect(ctx)
	t.Cleanup(func() { s.Disconnect(ctx) })

	s.OnTurnContextCompleted(ctx, "ctx-1")

	time.Sleep(50 * time.Millisecond)
	if sent := f.sent(); len(sent) != 0 {
		t.Errorf("sent %v, want nothing", sent)
	}
}

// A turn with text is flushed and then closed once its text has all been sent,
// and nothing cancels it.
func TestRealtimeTTSTurnWithTextIsFlushedAndClosed(t *testing.T) {
	f := newFakeInworld(t, &fakeInworld{})
	b, _ := testRealtimeTTS(f, RealtimeTTSConfig{})

	runTest(t, b, queue(frames.NewLLMFullResponseStartFrame()), queue(frames.NewLLMTextFrame("Hello there. ")),
		queue(frames.NewLLMFullResponseEndFrame()), wait(300*time.Millisecond))

	want := []string{"create", "send_text", "flush_context", "close_context"}
	if got := kinds(f.sent()); !slices.Equal(got, want) {
		t.Errorf("messages = %v, want %v", got, want)
	}
}

// An interruption cancels the context it cuts off, so the server stops
// synthesizing it.
func TestRealtimeTTSInterruptionCancelsContext(t *testing.T) {
	f := newFakeInworld(t, &fakeInworld{silent: true})
	b, _ := testRealtimeTTS(f, RealtimeTTSConfig{})

	runTest(t, b,
		queue(frames.NewLLMFullResponseStartFrame()),
		queue(frames.NewLLMTextFrame("Hello there. ")),
		queue(frames.NewLLMTextFrame("How are")),
		wait(100*time.Millisecond),
		queue(frames.NewInterruptionFrame()),
		wait(100*time.Millisecond))

	sent := f.sent()
	if got, want := kinds(sent), []string{"create", "send_text", "cancel_context"}; !slices.Equal(got, want) {
		t.Errorf("messages = %v, want %v", got, want)
	}
	if ids := contextIDs(sent); len(ids) != 1 {
		t.Errorf("contexts = %v, want one", ids)
	}
}

// A turn completed but still being spoken is closed, then canceled when the
// user interrupts it.
func TestRealtimeTTSInterruptionAfterTurnCompletedCancelsContext(t *testing.T) {
	f := newFakeInworld(t, &fakeInworld{silent: true})
	b, _ := testRealtimeTTS(f, RealtimeTTSConfig{})

	runTest(t, b,
		queue(frames.NewLLMFullResponseStartFrame()),
		queue(frames.NewLLMTextFrame("Hello there. How are you? ")),
		queue(frames.NewLLMFullResponseEndFrame()),
		wait(100*time.Millisecond),
		queue(frames.NewInterruptionFrame()),
		wait(100*time.Millisecond))

	sent := f.sent()
	got := kinds(sent)
	if len(got) < 2 || !slices.Equal(got[len(got)-2:], []string{"close_context", "cancel_context"}) {
		t.Errorf("messages = %v, want them to end with close_context, cancel_context", got)
	}
	if ids := contextIDs(sent); len(ids) != 1 {
		t.Errorf("contexts = %v, want one", ids)
	}
}

// Disconnecting cancels the contexts still open, so the server stops
// synthesizing them.
func TestRealtimeTTSDisconnectCancelsOpenContexts(t *testing.T) {
	f := newFakeInworld(t, &fakeInworld{silent: true})
	b, s := testRealtimeTTS(f, RealtimeTTSConfig{})
	ctx := context.Background()

	b.CreateAudioContext("ctx-1")
	if err := s.RunTTS(ctx, "Hello there. ", "ctx-1", nil); err != nil {
		t.Fatalf("run tts: %v", err)
	}
	_ = s.DisconnectWebsocket(ctx)

	sent := f.waitForMessages(t, 3)
	last := sent[len(sent)-1]
	if !reflect.DeepEqual(last, map[string]any{"cancel_context": map[string]any{}, "contextId": "ctx-1"}) {
		t.Errorf("last message = %v, want the context canceled", last)
	}
}

// step is one thing a test does to the running pipeline: queue a frame, or
// wait.
type step struct {
	frame frames.Frame
	sleep time.Duration
}

func queue(f frames.Frame) step { return step{frame: f} }
func wait(d time.Duration) step { return step{sleep: d} }

// runTest runs the service in a pipeline through the steps, ends the pipeline,
// and returns the frames that left it downstream and upstream.
func runTest(t *testing.T, svc *tts.Base, steps ...step) (down, up []frames.Frame) {
	t.Helper()
	var mu sync.Mutex
	task := pipeline.NewWorker(pipeline.New(svc), pipeline.WorkerConfig{
		ReachedDownstreamFilter: pipeline.AnyFrame,
		ReachedUpstreamFilter:   pipeline.AnyFrame,
	})
	events.On(&task.Registry, pipeline.EventFrameReachedDownstream, func(_ context.Context, f frames.Frame) {
		mu.Lock()
		down = append(down, f)
		mu.Unlock()
	})
	events.On(&task.Registry, pipeline.EventFrameReachedUpstream, func(_ context.Context, f frames.Frame) {
		mu.Lock()
		up = append(up, f)
		mu.Unlock()
	})
	done := make(chan error, 1)
	go func() { done <- task.Run(context.Background()) }()
	for _, s := range steps {
		if s.frame != nil {
			task.QueueFrame(s.frame)
		}
		if s.sleep > 0 {
			time.Sleep(s.sleep)
		}
	}
	task.StopWhenDone()
	select {
	case <-done:
	case <-time.After(10 * time.Second):
		t.Fatal("the pipeline did not finish")
	}
	mu.Lock()
	defer mu.Unlock()
	return slices.Clone(down), slices.Clone(up)
}

// speakTurns speaks each turn as a TTSSpeakFrame, half a second apart.
func speakTurns(texts ...string) []step {
	var out []step
	for _, text := range texts {
		out = append(out, queue(frames.NewTTSSpeakFrame(text)), wait(500*time.Millisecond))
	}
	return out
}

// turnsWithAudio is how many contexts produced audio.
func turnsWithAudio(down []frames.Frame) int {
	ids := map[string]bool{}
	for _, f := range down {
		if a, ok := f.(*frames.TTSAudioRawFrame); ok {
			ids[a.ContextID] = true
		}
	}
	return len(ids)
}

// errorTexts is the text of every error frame.
func errorTexts(fs []frames.Frame) []string {
	var out []string
	for _, f := range fs {
		if e, ok := f.(*frames.ErrorFrame); ok {
			out = append(out, e.Error)
		}
	}
	return out
}

// fastFailures makes a stalled send fail fast and a reconnect start at once.
func fastFailures() wsservice.Options {
	sendTimeout := 100 * time.Millisecond
	return wsservice.Options{
		SendTimeout:             &sendTimeout,
		ReconnectBackoffMinWait: 50 * time.Millisecond,
		ReconnectBackoffMaxWait: 50 * time.Millisecond,
	}
}

// stallSends makes the first send matching stall the way a send does once the
// peer stops reading: it waits out the send timeout, then the connection is
// dropped.
func stallSends(s *realtimeSynthesizer, timeout time.Duration, shouldStall func(map[string]any) bool) {
	write := s.write
	var mu sync.Mutex
	stalled := false
	s.write = func(ctx context.Context, conn *wsutil.Conn, payload []byte) error {
		var msg map[string]any
		_ = json.Unmarshal(payload, &msg)
		mu.Lock()
		stall := !stalled && shouldStall(msg)
		if stall {
			stalled = true
		}
		mu.Unlock()
		if stall {
			time.Sleep(timeout)
			_ = conn.CloseNow()
			return fmt.Errorf("%w within %s", wsutil.ErrSendTimeout, timeout)
		}
		return write(ctx, conn, payload)
	}
}

// sendsText matches the message sending text.
func sendsText(text string) func(map[string]any) bool {
	return func(m map[string]any) bool {
		st, ok := m["send_text"].(map[string]any)
		if !ok {
			return false
		}
		got, _ := st["text"].(string)
		return strings.TrimSpace(got) == text
	}
}

// A connection dropped after a turn is replaced before the next one, which is
// spoken on the new connection.
func TestRealtimeTTSADroppedConnectionIsReplacedBeforeTheNextTurn(t *testing.T) {
	f := newFakeInworld(t, &fakeInworld{dropAfterTurns: 1})
	b, _ := testRealtimeTTS(f, RealtimeTTSConfig{})

	down, _ := runTest(t, b, speakTurns(turns[:2]...)...)

	if n := f.handshakeCount(); n != 2 {
		t.Errorf("handshakes = %d, want 2", n)
	}
	if got := f.textsSent(1); !slices.Equal(got, []string{turns[1]}) {
		t.Errorf("texts on the second connection = %q, want %q", got, turns[1:2])
	}
	if n := turnsWithAudio(down); n != 2 {
		t.Errorf("turns with audio = %d, want 2", n)
	}
}

// A send that stalls fails its turn and drops the connection; the next turn is
// spoken on a new one, and the service stays usable.
func TestRealtimeTTSAStalledSendFailsItsTurnAndTheNextTurnIsSpoken(t *testing.T) {
	f := newFakeInworld(t, &fakeInworld{})
	b, s := testRealtimeTTS(f, RealtimeTTSConfig{Websocket: fastFailures()})
	stallSends(s, 100*time.Millisecond, sendsText(turns[1]))

	down, up := runTest(t, b, speakTurns(turns...)...)

	if n := f.handshakeCount(); n != 2 {
		t.Errorf("handshakes = %d, want 2", n)
	}
	if got := f.textsSent(1); !slices.Equal(got, []string{turns[2]}) {
		t.Errorf("texts on the second connection = %q, want %q", got, turns[2:])
	}
	if n := turnsWithAudio(down); n != 2 {
		t.Errorf("turns with audio = %d, want 2", n)
	}
	if len(errorTexts(up)) == 0 {
		t.Error("no error was reported for the failed turn")
	}
	if !b.Usable() {
		t.Error("the service is no longer usable")
	}
}

// A turn starting while the receive loop is reconnecting is skipped with an
// error rather than opening a second connection that nothing reads.
func TestRealtimeTTSATurnStartingWhileReconnectingDoesNotOpenASecondConnection(t *testing.T) {
	// The second turn's create stalls and drops the connection. The slow
	// handshake keeps the receive loop reconnecting when the third turn starts.
	f := newFakeInworld(t, &fakeInworld{handshakeDelay: time.Second})
	b, s := testRealtimeTTS(f, RealtimeTTSConfig{Websocket: fastFailures()})
	var mu sync.Mutex
	creates := 0
	stallSends(s, 100*time.Millisecond, func(m map[string]any) bool {
		if m["create"] == nil {
			return false
		}
		mu.Lock()
		defer mu.Unlock()
		creates++
		return creates == 2
	})

	down, up := runTest(t, b, speakTurns(turns...)...)

	if n := f.handshakeCount(); n != 2 {
		t.Errorf("handshakes = %d, want 2", n)
	}
	if n := turnsWithAudio(down); n != 1 {
		t.Errorf("turns with audio = %d, want 1", n)
	}
	if !slices.ContainsFunc(errorTexts(up), func(e string) bool { return strings.Contains(e, "reconnecting") }) {
		t.Errorf("errors = %q, want one saying the service is reconnecting", errorTexts(up))
	}
}

func TestRealtimeTTSExpressionAdapterFollowsTheModel(t *testing.T) {
	for _, tt := range []struct {
		model   string
		adapter ttsadapter.ExpressionAdapter
	}{
		{"inworld-tts-2", ttsinworld.InworldExpressionAdapter{}},
		{"inworld-tts-2-flash", ttsinworld.InworldFlashExpressionAdapter{}},
		{"inworld-tts-1.5-max", ttsadapter.BaseExpressionAdapter{}},
	} {
		_, s := newRealtimeTTS(RealtimeTTSConfig{APIKey: "key", Model: tt.model})
		if got := s.ExpressionAdapter(); reflect.TypeOf(got) != reflect.TypeOf(tt.adapter) {
			t.Errorf("%s: adapter %T, want %T", tt.model, got, tt.adapter)
		}
	}
}

// A pause tag is taken out of the word Inworld reports it as.
func TestRealtimeTTSAPauseIsLeftOutOfTheWordsReported(t *testing.T) {
	_, s := newRealtimeTTS(RealtimeTTSConfig{APIKey: "key"})
	info := &timestampInfo{}
	info.WordAlignment.Words = []string{"there", ` <break time="500ms"/> `, "friend"}
	info.WordAlignment.WordStartTimeSeconds = []float64{0.1, 0.4, 0.9}
	info.WordAlignment.WordEndTimeSeconds = []float64{0.3, 0.9, 1.2}

	var words []string
	for _, w := range s.calculateWordTimes(info) {
		words = append(words, w.Word)
	}

	if want := []string{"there", "  ", "friend"}; !slices.Equal(words, want) {
		t.Errorf("words = %q, want %q", words, want)
	}
}

func TestRealtimeTTSFormatsPronunciationsLikeHTTP(t *testing.T) {
	_, s := newRealtimeTTS(RealtimeTTSConfig{APIKey: "key"})
	got, _ := s.FormatPronunciation("Crete", "kriːt")
	want, _ := (&synthesizer{}).FormatPronunciation("Crete", "kriːt")
	if got != want {
		t.Errorf("the WebSocket service formats %q, the HTTP one %q", got, want)
	}
}

func TestInworldBaseLanguagesResolveToCanonicalRegionalTags(t *testing.T) {
	for l, want := range map[language.Language]string{
		language.English: "en-US",
		language.Russian: "ru-RU",
		language.French:  "fr-FR",
		language.Chinese: "zh-CN",
	} {
		if got := inworldLanguage(l); got != want {
			t.Errorf("%s named %q, want %q", l, got, want)
		}
	}
}

func TestInworldRegionalLanguagesArePreserved(t *testing.T) {
	for l, want := range map[language.Language]string{
		language.EnglishGB:    "en-GB",
		language.PortuguesePT: "pt-PT",
		language.RussianRU:    "ru-RU",
	} {
		if got := inworldLanguage(l); got != want {
			t.Errorf("%s named %q, want %q", l, got, want)
		}
	}
}

func TestInworldOtherLanguagesArePassedThroughAsBCP47Tags(t *testing.T) {
	for l, want := range map[language.Language]string{
		language.SwedishSE:   "sv-SE",
		language.UkrainianUA: "uk-UA",
	} {
		if got := inworldLanguage(l); got != want {
			t.Errorf("%s named %q, want %q", l, got, want)
		}
	}
}
