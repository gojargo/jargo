package bland

import (
	"bytes"
	"context"
	"encoding/binary"
	"encoding/json"
	"maps"
	"net/http"
	"net/http/httptest"
	"reflect"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/coder/websocket"
	"github.com/gojargo/jargo/event"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/service"
	"github.com/gojargo/jargo/service/settings"
	"github.com/gojargo/jargo/service/tts"
)

const otherVoiceID = "c18a1cd5-91ef-4b06-841a-e58b8b487e8c"

//nolint:gochecknoglobals // test fixtures, read-only
var (
	audioChunk1 = bytes.Repeat([]byte{0x00, 0x01}, 512)
	audioChunk2 = bytes.Repeat([]byte{0x02, 0x03}, 512)
)

// pcmBytes is bare little-endian int16 PCM, which is what container raw
// returns.
func pcmBytes(numSamples int) []byte {
	out := make([]byte, 0, numSamples*2)
	for i := range numSamples {
		out = binary.LittleEndian.AppendUint16(out, uint16(int16((i*97)%2000-1000))) //nolint:gosec // a test signal
	}
	return out
}

// audioOf joins the audio of the frames.
func audioOf(fs []frames.Frame) []byte {
	var out []byte
	for _, f := range fs {
		if a, ok := f.(*frames.TTSAudioRawFrame); ok {
			out = append(out, a.Audio...)
		}
	}
	return out
}

// errorsOf is the messages of the error frames.
func errorsOf(fs []frames.Frame) []string {
	var out []string
	for _, f := range fs {
		if ef, ok := f.(*frames.ErrorFrame); ok {
			out = append(out, ef.Error)
		}
	}
	return out
}

func hasFrame[T frames.Frame](fs []frames.Frame) bool {
	return slices.ContainsFunc(fs, func(f frames.Frame) bool { _, ok := f.(T); return ok })
}

// sampleRateOf is the sample rate a message's audio format asks for.
func sampleRateOf(msg map[string]any) any {
	audio, _ := msg["audio"].(map[string]any)
	return audio["sample_rate"]
}

// --- /v2/tts/ws ---

// session is one connection to the fake server, as its handler sees it.
type session struct {
	ctx    context.Context
	conn   *websocket.Conn
	number int
	// messages are the messages this session received.
	messages []map[string]any
}

func (s *session) sendJSON(msg map[string]any) {
	b, _ := json.Marshal(msg)
	_ = s.conn.Write(s.ctx, websocket.MessageText, b)
}

func (s *session) sendBinary(b []byte) { _ = s.conn.Write(s.ctx, websocket.MessageBinary, b) }

func (s *session) close(code websocket.StatusCode, reason string) { _ = s.conn.Close(code, reason) }

// readyFor acknowledges an init message at the rate it asked for.
func (s *session) readyFor(msg map[string]any) {
	rate := 48000.0
	if audio, ok := msg["audio"].(map[string]any); ok {
		if r, ok := audio["sample_rate"].(float64); ok {
			rate = r
		}
	}
	s.sendJSON(map[string]any{"type": "ready", "session_id": "s1", "encoding": "pcm_s16le", "sample_rate": rate})
}

// fakeBland is a fake Bland realtime server, running handle on each message
// a session receives until it returns false.
type fakeBland struct {
	url    string
	handle func(s *session, msg map[string]any) bool

	mu         sync.Mutex
	messages   []map[string]any
	sessions   [][]map[string]any
	authHeader string
	doneSent   bool
	closed     int
}

func newFakeBland(t *testing.T, handle func(s *session, msg map[string]any) bool) *fakeBland {
	t.Helper()
	f := &fakeBland{handle: handle}
	srv := httptest.NewServer(http.HandlerFunc(f.serve))
	t.Cleanup(srv.Close)
	f.url = "ws" + strings.TrimPrefix(srv.URL, "http") + "/v2/tts/ws"
	return f
}

func (f *fakeBland) serve(w http.ResponseWriter, r *http.Request) {
	c, err := websocket.Accept(w, r, nil)
	if err != nil {
		return
	}
	defer func() { _ = c.CloseNow() }()
	f.mu.Lock()
	f.authHeader = r.Header.Get("Authorization")
	f.sessions = append(f.sessions, nil)
	s := &session{ctx: r.Context(), conn: c, number: len(f.sessions)}
	f.mu.Unlock()
	for {
		_, data, err := c.Read(s.ctx)
		if err != nil {
			f.mu.Lock()
			f.closed++
			f.mu.Unlock()
			return
		}
		var msg map[string]any
		if json.Unmarshal(data, &msg) != nil {
			return
		}
		f.mu.Lock()
		f.messages = append(f.messages, msg)
		f.sessions[s.number-1] = append(f.sessions[s.number-1], msg)
		f.mu.Unlock()
		s.messages = append(s.messages, msg)
		if !f.handle(s, msg) {
			return
		}
	}
}

// ofType is the messages of a type the server received, across sessions.
func (f *fakeBland) ofType(kind string) []map[string]any {
	f.mu.Lock()
	defer f.mu.Unlock()
	var out []map[string]any
	for _, m := range f.messages {
		if m["type"] == kind {
			out = append(out, m)
		}
	}
	return out
}

func (f *fakeBland) sessionCount() int {
	f.mu.Lock()
	defer f.mu.Unlock()
	return len(f.sessions)
}

// turnFlow is how the documented server behaves, and the ways it can be made
// to misbehave.
type turnFlow struct {
	initError       map[string]any
	turnError       map[string]any
	endReason       string
	readyEncoding   string
	readySampleRate float64
	noAcknowledge   bool
}

// documented builds a fake server following the documented turn flow.
func documented(t *testing.T, flow turnFlow) *fakeBland {
	t.Helper()
	if flow.endReason == "" {
		flow.endReason = "complete"
	}
	if flow.readyEncoding == "" {
		flow.readyEncoding = "pcm_s16le"
	}
	started := map[string]bool{}
	var f *fakeBland
	f = newFakeBland(t, func(s *session, msg map[string]any) bool {
		switch msg["type"] {
		case "init":
			if flow.initError != nil {
				reply := map[string]any{"type": "error"}
				maps.Copy(reply, flow.initError)
				s.sendJSON(reply)
				s.close(websocket.StatusNormalClosure, "")
				return false
			}
			if flow.noAcknowledge {
				return true
			}
			rate := flow.readySampleRate
			if rate == 0 {
				rate, _ = sampleRateOf(msg).(float64)
			}
			s.sendJSON(map[string]any{
				"type": "ready", "session_id": "test-session", "encoding": flow.readyEncoding, "sample_rate": rate,
			})
		case "speak":
			id := msg["context_id"]
			if flow.turnError != nil {
				reply := map[string]any{"type": "error", "context_id": id}
				maps.Copy(reply, flow.turnError)
				s.sendJSON(reply)
				return true
			}
			if key, _ := id.(string); !started[key] {
				started[key] = true
				s.sendJSON(map[string]any{"type": "utterance_start", "context_id": id})
			}
		case "end_of_turn":
			if flow.turnError != nil {
				return true
			}
			s.sendBinary(audioChunk1)
			s.sendBinary(audioChunk2)
			s.sendJSON(map[string]any{
				"type": "utterance_end", "context_id": msg["context_id"], "reason": flow.endReason,
				"frames": 2, "duration_ms": 100,
			})
		case "close":
			s.sendJSON(map[string]any{"type": "done", "session_id": "test-session"})
			f.mu.Lock()
			f.doneSent = true
			f.mu.Unlock()
			s.close(websocket.StatusNormalClosure, "")
			return false
		}
		return true
	})
	return f
}

// step is one thing runTest does: queue a frame, or wait.
type step struct {
	frame frames.Frame
	sleep time.Duration
}

func queue(f frames.Frame) step { return step{frame: f} }
func wait(d time.Duration) step { return step{sleep: d} }

// speak speaks text, then waits for its audio and stops the bot speaking.
func speak(text string) []step {
	return []step{
		queue(frames.NewTTSSpeakFrame(text)),
		wait(300 * time.Millisecond),
		queue(frames.NewBotStoppedSpeakingFrame()),
	}
}

// runTest runs the service in a pipeline through the steps, ends the pipeline,
// and returns the frames that left it downstream and upstream.
func runTest(t *testing.T, svc *tts.Base, ss ...step) (down, up []frames.Frame) {
	t.Helper()
	var mu sync.Mutex
	task := pipeline.NewWorker(pipeline.New(svc), pipeline.WorkerConfig{
		ReachedDownstreamFilter: pipeline.AnyFrame,
		ReachedUpstreamFilter:   pipeline.AnyFrame,
	})
	event.On(&task.Registry, pipeline.EventFrameReachedDownstream, func(_ context.Context, f frames.Frame) {
		mu.Lock()
		down = append(down, f)
		mu.Unlock()
	})
	event.On(&task.Registry, pipeline.EventFrameReachedUpstream, func(_ context.Context, f frames.Frame) {
		mu.Lock()
		up = append(up, f)
		mu.Unlock()
	})
	done := make(chan error, 1)
	go func() { done <- task.Run(context.Background()) }()
	for _, s := range ss {
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
	case <-time.After(15 * time.Second):
		t.Fatal("the pipeline did not finish")
	}
	mu.Lock()
	defer mu.Unlock()
	return slices.Clone(down), slices.Clone(up)
}

// newService builds the realtime service against the fake server at 24 kHz.
func newService(f *fakeBland, cfg Config) *tts.Base {
	cfg.APIKey = "test-key"
	cfg.URL = f.url
	if cfg.SampleRate == 0 {
		cfg.SampleRate = 24000
	}
	return NewTTS(cfg)
}

// init, speak and end_of_turn are sent, and the turn's audio is emitted.
func TestProtocolRoundtrip(t *testing.T) {
	f := documented(t, turnFlow{})
	down, up := runTest(t, newService(f, Config{}), speak("Hello from Bland.")...)

	if !hasFrame[*frames.TTSStartedFrame](down) || !hasFrame[*frames.TTSAudioRawFrame](down) ||
		!hasFrame[*frames.TTSStoppedFrame](down) {
		t.Errorf("frames = %v, want started, audio and stopped", down)
	}
	if errs := errorsOf(slices.Concat(down, up)); len(errs) > 0 {
		t.Errorf("errors = %v, want none", errs)
	}
	for _, fr := range down {
		if a, ok := fr.(*frames.TTSAudioRawFrame); ok && (a.SampleRate != 24000 || a.NumChannels != 1) {
			t.Errorf("audio at %d Hz on %d channels, want 24 kHz mono", a.SampleRate, a.NumChannels)
		}
	}
	if got := audioOf(down); !bytes.Equal(got, slices.Concat(audioChunk1, audioChunk2)) {
		t.Errorf("audio = %d bytes, want the two chunks", len(got))
	}
	if f.authHeader != "Bearer test-key" {
		t.Errorf("Authorization = %q", f.authHeader)
	}
	init := f.ofType("init")[0]
	if init["voice"] != defaultVoiceID {
		t.Errorf("voice = %v, want the default", init["voice"])
	}
	if !reflect.DeepEqual(init["audio"], map[string]any{"encoding": "pcm_s16le", "sample_rate": float64(24000)}) {
		t.Errorf("audio = %v", init["audio"])
	}
	if _, ok := init["controls"]; ok {
		t.Error("controls were sent")
	}
	if _, ok := init["auto_formatting"]; ok {
		t.Error("auto_formatting was sent")
	}
	speakMsg := f.ofType("speak")[0]
	endOfTurn := f.ofType("end_of_turn")[0]
	if speakMsg["text"] != "Hello from Bland." {
		t.Errorf("speak text = %v", speakMsg["text"])
	}
	// The turn is ended under the id its deltas were sent with.
	if endOfTurn["context_id"] != speakMsg["context_id"] {
		t.Errorf("end_of_turn on %v, speak on %v", endOfTurn["context_id"], speakMsg["context_id"])
	}
}

// In the default token mode, the model's tokens map one to one to speak
// messages, unaltered.
func TestTokenStreamingSendsTokensVerbatim(t *testing.T) {
	f := documented(t, turnFlow{})
	down, up := runTest(t, newService(f, Config{}),
		queue(frames.NewLLMFullResponseStartFrame()),
		queue(frames.NewLLMTextFrame("Unbelieva")),
		queue(frames.NewLLMTextFrame("ble")),
		queue(frames.NewLLMTextFrame(" isn't it?")),
		queue(frames.NewLLMFullResponseEndFrame()),
		wait(300*time.Millisecond),
		queue(frames.NewBotStoppedSpeakingFrame()))

	if errs := errorsOf(slices.Concat(down, up)); len(errs) > 0 {
		t.Errorf("errors = %v, want none", errs)
	}
	if !hasFrame[*frames.TTSAudioRawFrame](down) {
		t.Error("no audio")
	}
	speaks := f.ofType("speak")
	var texts []string
	ids := map[any]bool{}
	for _, m := range speaks {
		text, _ := m["text"].(string)
		texts = append(texts, text)
		ids[m["context_id"]] = true
	}
	// Bland appends each delta verbatim, so an inserted space would split words.
	if want := []string{"Unbelieva", "ble", " isn't it?"}; !slices.Equal(texts, want) {
		t.Errorf("speak texts = %q, want %q", texts, want)
	}
	// Every delta of one response belongs to one turn, ended once.
	if len(ids) != 1 || len(f.ofType("end_of_turn")) != 1 {
		t.Errorf("contexts = %v, end_of_turn = %d, want one of each", ids, len(f.ofType("end_of_turn")))
	}
}

// In sentence mode a trailing space separates consecutive generations.
func TestSentenceModeAppendsTrailingSpace(t *testing.T) {
	f := documented(t, turnFlow{})
	down, up := runTest(t, newService(f, Config{TextAggregation: frames.TextTypeSentence}), speak("Hello from Bland.")...)

	if errs := errorsOf(slices.Concat(down, up)); len(errs) > 0 {
		t.Errorf("errors = %v, want none", errs)
	}
	if got := f.ofType("speak")[0]["text"]; got != "Hello from Bland. " {
		t.Errorf("speak text = %q, want a trailing space", got)
	}
}

// The voice and the controls are fixed at init for the life of the session.
func TestInitCarriesControls(t *testing.T) {
	f := documented(t, turnFlow{})
	expressiveness, stability := 0.9, 0.4
	runTest(t, newService(f, Config{Voice: otherVoiceID, Expressiveness: &expressiveness, Stability: &stability}))

	init := f.ofType("init")[0]
	if init["voice"] != otherVoiceID {
		t.Errorf("voice = %v", init["voice"])
	}
	if !reflect.DeepEqual(init["controls"], map[string]any{"expressiveness": 0.9, "stability": 0.4}) {
		t.Errorf("controls = %v", init["controls"])
	}
}

// Only the controls the caller set are sent, so unset ones keep Bland's
// defaults.
func TestPartialControls(t *testing.T) {
	f := documented(t, turnFlow{})
	stability := 0.4
	runTest(t, newService(f, Config{Stability: &stability}))

	if got := f.ofType("init")[0]["controls"]; !reflect.DeepEqual(got, map[string]any{"stability": 0.4}) {
		t.Errorf("controls = %v", got)
	}
}

// auto_formatting is fixed at init, like the controls.
func TestInitCarriesAutoFormatting(t *testing.T) {
	f := documented(t, turnFlow{})
	runTest(t, newService(f, Config{AutoFormatting: new(true)}))

	init := f.ofType("init")[0]
	if init["auto_formatting"] != true {
		t.Errorf("auto_formatting = %v", init["auto_formatting"])
	}
	if _, ok := init["controls"]; ok {
		t.Error("controls were sent")
	}
}

// Changing auto_formatting opens a new session, since init fixes it.
func TestAutoFormattingUpdateReconnects(t *testing.T) {
	f := documented(t, turnFlow{})
	delta := &TTSSettings{AutoFormatting: settings.Set(true)}
	runTest(t, newService(f, Config{}), queue(frames.NewTTSUpdateSettingsFrame(delta)), wait(300*time.Millisecond))

	inits := f.ofType("init")
	if len(inits) != 2 {
		t.Fatalf("inits = %d, want 2", len(inits))
	}
	if _, ok := inits[0]["auto_formatting"]; ok {
		t.Error("the first session carried auto_formatting")
	}
	if inits[1]["auto_formatting"] != true {
		t.Errorf("auto_formatting = %v, want true on the new session", inits[1]["auto_formatting"])
	}
}

// A rate Bland cannot render is replaced by its native 48 kHz, and the audio is
// resampled to the service's rate.
func TestUnsupportedPipelineRateFallsBack(t *testing.T) {
	f := documented(t, turnFlow{})
	down, _ := runTest(t, newService(f, Config{SampleRate: 22050}), speak("Hi.")...)

	if got := sampleRateOf(f.ofType("init")[0]); got != float64(48000) {
		t.Errorf("init sample_rate = %v, want 48000", got)
	}
	var audio []*frames.TTSAudioRawFrame
	for _, fr := range down {
		if a, ok := fr.(*frames.TTSAudioRawFrame); ok {
			audio = append(audio, a)
		}
	}
	if len(audio) == 0 {
		t.Fatal("no audio")
	}
	for _, a := range audio {
		if a.SampleRate != 22050 {
			t.Errorf("audio at %d Hz, want it resampled to 22050", a.SampleRate)
		}
	}
}

// Barge-in sends cancel, so the session and its warm voice survive.
func TestInterruptionCancelsWithoutReconnecting(t *testing.T) {
	f := documented(t, turnFlow{})
	_, s := newTTS(Config{APIKey: "test-key", URL: f.url, SampleRate: 24000})
	ctx := context.Background()
	s.Connect(ctx)
	t.Cleanup(func() { s.Disconnect(ctx) })

	s.OnAudioContextInterrupted(ctx, "turn-17")

	waitUntil(t, func() bool { return len(f.ofType("cancel")) == 1 })
	f.mu.Lock()
	sent := slices.Clone(f.messages[1:])
	f.mu.Unlock()
	if want := []map[string]any{{"type": "cancel", "context_id": "turn-17"}}; !reflect.DeepEqual(sent, want) {
		t.Errorf("sent %v, want %v", sent, want)
	}
	if !s.Connected() || f.sessionCount() != 1 {
		t.Error("the interruption closed the session")
	}
}

// A turn-scoped error becomes an ErrorFrame carrying its code and message, and
// ends the turn.
func TestTurnErrorSurfaces(t *testing.T) {
	for _, code := range []string{"insufficient_credits", "rate_limited"} {
		t.Run(code, func(t *testing.T) {
			f := documented(t, turnFlow{turnError: map[string]any{"code": code, "message": "Turn admission refused."}})
			b := newService(f, Config{})
			down, up := runTest(t, b, speak("Hi.")...)

			errs := errorsOf(slices.Concat(down, up))
			if len(errs) == 0 || !strings.Contains(errs[0], code) || !strings.Contains(errs[0], "Turn admission refused.") {
				t.Errorf("errors = %v, want the turn's", errs)
			}
			if ids := b.AudioContextIDs(); len(ids) != 0 {
				t.Errorf("audio contexts = %v, want none", ids)
			}
		})
	}
}

// A turn that ends as failed, or that the server preempted, is closed rather
// than left waiting for audio.
func TestATurnTheServerEndedIsClosed(t *testing.T) {
	for _, reason := range []string{"failed", "preempted"} {
		t.Run(reason, func(t *testing.T) {
			f := documented(t, turnFlow{endReason: reason})
			b := newService(f, Config{})
			down, _ := runTest(t, b, speak("Hi.")...)

			if !hasFrame[*frames.TTSStoppedFrame](down) {
				t.Error("no stop frame")
			}
			if ids := b.AudioContextIDs(); len(ids) != 0 {
				t.Errorf("audio contexts = %v, want none", ids)
			}
		})
	}
}

// A session Bland refuses fails at connect, not on the first turn.
func TestRejectedInitSurfaces(t *testing.T) {
	f := documented(t, turnFlow{initError: map[string]any{"code": "voice_not_found", "message": "Voice was not found."}})
	down, up := runTest(t, newService(f, Config{}))

	errs := errorsOf(slices.Concat(down, up))
	if len(errs) == 0 || !strings.Contains(errs[0], "voice_not_found") {
		t.Errorf("errors = %v, want the refusal", errs)
	}
}

// A peer that upgrades but never acknowledges init cannot hang startup, and
// the connection it made is closed.
func TestInitTimeoutClosesProvisionalConnection(t *testing.T) {
	f := documented(t, turnFlow{noAcknowledge: true})
	b, s := newTTS(Config{APIKey: "test-key", URL: f.url, SampleRate: 24000})
	s.readyTimeout = 50 * time.Millisecond
	down, up := runTest(t, b)

	if errs := errorsOf(slices.Concat(down, up)); len(errs) == 0 {
		t.Error("the unacknowledged session was not reported")
	}
	waitUntil(t, func() bool {
		f.mu.Lock()
		defer f.mu.Unlock()
		return f.closed > 0
	})
}

// A connect canceled while waiting for ready closes the connection it made, and
// is not reported.
func TestCanceledInitClosesProvisionalConnection(t *testing.T) {
	f := documented(t, turnFlow{noAcknowledge: true})
	b, s := newTTS(Config{APIKey: "test-key", URL: f.url, SampleRate: 24000})
	var reported bool
	event.On(b.Events(), service.EventConnectionError, func(context.Context, string) { reported = true })
	ctx, cancel := context.WithCancel(context.Background())
	time.AfterFunc(50*time.Millisecond, cancel)

	if err := s.ConnectWebsocket(ctx); err != nil {
		t.Fatal(err)
	}
	if s.Connected() || reported {
		t.Errorf("connected = %t, reported = %t, want neither", s.Connected(), reported)
	}
	waitUntil(t, func() bool {
		f.mu.Lock()
		defer f.mu.Unlock()
		return f.closed > 0
	})
}

func TestAPIKeyIsRequired(t *testing.T) {
	if (Config{}).Validate() == nil {
		t.Error("the realtime service took an empty API key")
	}
	if (HTTPTTSConfig{}).Validate() == nil {
		t.Error("the HTTP service took an empty API key")
	}
	if (Config{APIKey: "k"}).Validate() != nil || (HTTPTTSConfig{APIKey: "k"}).Validate() != nil {
		t.Error("a key was refused")
	}
}

// Audio is never labeled with a format the server did not acknowledge.
func TestRejectsMismatchedReadyFormat(t *testing.T) {
	for name, flow := range map[string]turnFlow{
		"encoding": {readyEncoding: "mulaw", readySampleRate: 24000},
		"rate":     {readyEncoding: "pcm_s16le", readySampleRate: 16000},
	} {
		t.Run(name, func(t *testing.T) {
			f := documented(t, flow)
			down, up := runTest(t, newService(f, Config{}))
			errs := errorsOf(slices.Concat(down, up))
			if len(errs) == 0 || !strings.Contains(errs[0], "unexpected audio format") {
				t.Errorf("errors = %v, want the format refused", errs)
			}
		})
	}
}

// Shutdown asks Bland to settle usage instead of dropping the socket.
func TestCloseSettlesTheSession(t *testing.T) {
	f := documented(t, turnFlow{})
	runTest(t, newService(f, Config{}), speak("Hi.")...)

	if n := len(f.ofType("close")); n != 1 {
		t.Errorf("close sent %d times, want once", n)
	}
	f.mu.Lock()
	defer f.mu.Unlock()
	if !f.doneSent || len(f.sessions) != 1 {
		t.Errorf("done sent = %t over %d sessions, want it sent over one", f.doneSent, len(f.sessions))
	}
}

// --- turns that cannot finish ---

// refusing refuses admission for each turn's context once, as the server does,
// and drops that context's later deltas silently.
func refusing(t *testing.T) *fakeBland {
	t.Helper()
	refused := map[any]bool{}
	return newFakeBland(t, func(s *session, msg map[string]any) bool {
		switch msg["type"] {
		case "init":
			s.readyFor(msg)
		case "speak":
			id := msg["context_id"]
			if refused[id] {
				return true
			}
			refused[id] = true
			s.sendJSON(map[string]any{
				"type": "error", "context_id": id, "code": "insufficient_credits", "message": "wallet depleted",
			})
		case "close":
			s.sendJSON(map[string]any{"type": "done", "session_id": "s1"})
			return false
		}
		return true
	})
}

// A refused turn's error reaches the pipeline, and later deltas are still sent.
func TestReportsARefusedTurn(t *testing.T) {
	f := refusing(t)
	down, up := runTest(t, newService(f, Config{}),
		queue(frames.NewLLMFullResponseStartFrame()),
		queue(frames.NewLLMTextFrame("first")),
		wait(200*time.Millisecond),
		queue(frames.NewLLMTextFrame(" second")),
		wait(200*time.Millisecond),
		queue(frames.NewLLMTextFrame(" third")),
		queue(frames.NewLLMFullResponseEndFrame()),
		wait(200*time.Millisecond),
		queue(frames.NewBotStoppedSpeakingFrame()))

	var texts []string
	for _, m := range f.ofType("speak") {
		text, _ := m["text"].(string)
		texts = append(texts, text)
	}
	if want := []string{"first", " second", " third"}; !slices.Equal(texts, want) {
		t.Errorf("speak texts = %q, want %q", texts, want)
	}
	refusals := 0
	for _, e := range errorsOf(slices.Concat(down, up)) {
		if strings.Contains(e, "insufficient_credits") {
			refusals++
		}
	}
	if refusals != 1 {
		t.Errorf("refusals reported = %d, want 1", refusals)
	}
}

// Losing the socket midway through a turn sends the rest of the turn to the new
// session.
func TestContinuesATurnWhoseSocketDiedMidway(t *testing.T) {
	f := newFakeBland(t, func(s *session, msg map[string]any) bool {
		switch msg["type"] {
		case "init":
			s.readyFor(msg)
		case "speak":
			if s.number == 1 {
				s.close(websocket.StatusInternalError, "injected failure")
				return false
			}
		case "close":
			s.sendJSON(map[string]any{"type": "done", "session_id": "s"})
			return false
		}
		return true
	})
	runTest(t, newService(f, Config{}),
		queue(frames.NewLLMFullResponseStartFrame()),
		queue(frames.NewLLMTextFrame("The weather is clear")),
		wait(300*time.Millisecond),
		queue(frames.NewLLMTextFrame(" and warm today.")),
		queue(frames.NewLLMFullResponseEndFrame()),
		wait(300*time.Millisecond),
		queue(frames.NewBotStoppedSpeakingFrame()))

	f.mu.Lock()
	defer f.mu.Unlock()
	var first, later []map[string]any
	for i, msgs := range f.sessions {
		for _, m := range msgs {
			if m["type"] != "speak" {
				continue
			}
			if i == 0 {
				first = append(first, m)
			} else {
				later = append(later, m)
			}
		}
	}
	// The new session knows nothing of the old turn, so the rest of it goes out
	// as a new turn.
	if len(later) != 1 || later[0]["text"] != " and warm today." {
		t.Fatalf("later speaks = %v, want the rest of the turn", later)
	}
	if later[0]["context_id"] == first[0]["context_id"] {
		t.Error("the rest of the turn went out on the old context")
	}
}

// --- a turn the server has ended ---

// failingTurn ends each turn as failed, the way an oversized pause marker does,
// after an error carrying the detail when sendErrorFirst is set.
func failingTurn(t *testing.T, sendErrorFirst bool) *fakeBland {
	t.Helper()
	failed := map[any]bool{}
	return newFakeBland(t, func(s *session, msg map[string]any) bool {
		switch msg["type"] {
		case "init":
			s.readyFor(msg)
		case "speak":
			id := msg["context_id"]
			if failed[id] {
				return true
			}
			failed[id] = true
			s.sendJSON(map[string]any{"type": "utterance_start", "context_id": id})
			if sendErrorFirst {
				s.sendJSON(map[string]any{
					"type": "error", "context_id": id, "code": "invalid_request",
					"message": "Pause marker `<|30|>` exceeds the maximum.",
				})
			}
			s.sendJSON(map[string]any{
				"type": "utterance_end", "context_id": id, "reason": "failed", "frames": 0, "duration_ms": 1,
			})
		case "close":
			s.sendJSON(map[string]any{"type": "done", "session_id": "s1"})
			return false
		}
		return true
	})
}

// The error message carries the detail; the end of the turn must not add a
// vaguer one.
func TestReportsAFailedTurnOnce(t *testing.T) {
	f := failingTurn(t, true)
	down, up := runTest(t, newService(f, Config{}), speak("Hold on <|30|> there.")...)

	// The context separately reports completing with no audio; the turn itself
	// is reported once.
	var turnErrors []string
	for _, e := range errorsOf(slices.Concat(down, up)) {
		if !strings.Contains(e, "completed with no audio") {
			turnErrors = append(turnErrors, e)
		}
	}
	if len(turnErrors) != 1 || !strings.Contains(turnErrors[0], "invalid_request") {
		t.Errorf("turn errors = %q, want the one invalid_request", turnErrors)
	}
}

// A bare failed end closes the turn.
func TestClosesAFailedTurnWithNoErrorFrame(t *testing.T) {
	f := failingTurn(t, false)
	b := newService(f, Config{})
	down, _ := runTest(t, b, speak("Hi.")...)

	if !hasFrame[*frames.TTSStoppedFrame](down) {
		t.Error("no stop frame")
	}
	if ids := b.AudioContextIDs(); len(ids) != 0 {
		t.Errorf("audio contexts = %v, want none", ids)
	}
}

// Bland reaps an idle session itself; the session that replaces it is
// routine.
func TestIdleCloseIsNotAPipelineError(t *testing.T) {
	var reaped bool
	f := newFakeBland(t, func(s *session, msg map[string]any) bool {
		if msg["type"] != "init" {
			return true
		}
		s.readyFor(msg)
		// Reap the first session the way the 60s idle timeout does, and hold the
		// one that replaces it open.
		if !reaped {
			reaped = true
			s.sendJSON(map[string]any{"type": "error", "code": "idle_timeout", "message": "Session idle for 60s."})
			s.close(websocket.StatusInternalError, "idle")
			return false
		}
		return true
	})
	down, up := runTest(t, newService(f, Config{}), wait(400*time.Millisecond))

	// A reaped session is replaced, and nothing about that is the application's
	// problem to hear about.
	if n := len(f.ofType("init")); n != 2 {
		t.Errorf("inits = %d, want 2", n)
	}
	if errs := errorsOf(slices.Concat(down, up)); len(errs) > 0 {
		t.Errorf("errors = %v, want none", errs)
	}
}

// waitUntil waits for cond, failing the test if it never holds.
func waitUntil(t *testing.T, cond func() bool) {
	t.Helper()
	deadline := time.Now().Add(3 * time.Second)
	for !cond() {
		if time.Now().After(deadline) {
			t.Fatal("the condition never held")
		}
		time.Sleep(10 * time.Millisecond)
	}
}
